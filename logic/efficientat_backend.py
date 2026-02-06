import os
import sys
from typing import List, Optional

import numpy as np
import onnxruntime as ort

from .utils import get_user_data_path


class EfficientATBackend:
    """
    EfficientAT backend:
    - One-time exports EfficientAT (mn10_as) to ONNX.
    - Uses ONNX Runtime DirectML only for model inference.
    - Keeps mel preprocessing in torch CPU to match training preprocessing.
    """

    def __init__(self, model_name: str = "mn10_as", onnx_batch_size: int = 32):
        self.model_name = model_name
        self.sample_rate = 32000
        self.window_samples = self.sample_rate * 10
        self.n_mels = 128
        self.onnx_batch_size = max(1, int(onnx_batch_size))

        self.labels: List[str] = []
        self.ready = False
        self.load_error: Optional[str] = None
        self.device_name = "dml"

        self._torch = None
        self._mel = None
        self._session = None
        self._input_name = None
        self._onnx_path = None
        self._model_cache_dir = None

        try:
            self._prepare_paths_and_imports()
            self._init_mel_preprocessor()
            self._ensure_onnx_model()
            self._init_session()
            self.ready = True
        except Exception as e:
            self.ready = False
            self.load_error = str(e)

    def _prepare_paths_and_imports(self):
        third_party_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "third_party",
            "efficientat",
        )
        if third_party_root not in sys.path:
            sys.path.insert(0, third_party_root)

        base_cache = get_user_data_path("efficientat_models")
        onnx_cache = os.path.join(base_cache, "onnx")
        weights_cache = os.path.join(base_cache, "weights")
        os.makedirs(onnx_cache, exist_ok=True)
        os.makedirs(weights_cache, exist_ok=True)
        os.environ.setdefault("EFFICIENTAT_MODEL_DIR", weights_cache)

        self._model_cache_dir = base_cache
        self._onnx_path = os.path.join(
            onnx_cache,
            f"efficientat_{self.model_name}_b{self.onnx_batch_size}.onnx",
        )

        import torch
        from helpers.utils import labels

        self._torch = torch
        self.labels = list(labels)

    def _init_mel_preprocessor(self):
        from models.preprocess import AugmentMelSTFT

        self._mel = AugmentMelSTFT(
            n_mels=self.n_mels,
            sr=self.sample_rate,
            win_length=800,
            hopsize=320,
            fmax=15000,
            freqm=0,
            timem=0,
        )
        self._mel.eval()

    def _ensure_onnx_model(self):
        if self._onnx_path and os.path.exists(self._onnx_path):
            return

        torch = self._torch
        if torch is None:
            raise RuntimeError("Torch is required for EfficientAT ONNX export.")

        from helpers.utils import NAME_TO_WIDTH
        from models.mn.model import get_model as get_mn

        class _SigmoidWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                logits, _ = self.model(x)
                return torch.sigmoid(logits)

        width = NAME_TO_WIDTH(self.model_name)
        model = get_mn(
            width_mult=width,
            pretrained_name=self.model_name,
            head_type="mlp",
        )
        model.eval()
        wrapped = _SigmoidWrapper(model).eval()

        dummy = torch.randn(self.onnx_batch_size, 1, self.n_mels, 1000, dtype=torch.float32)
        torch.onnx.export(
            wrapped,
            dummy,
            self._onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,  # avoid cp932 logging issues in the new exporter on Windows terminals
        )

    def _init_session(self):
        if not self._onnx_path:
            raise RuntimeError("EfficientAT ONNX path is not initialized.")

        available = ort.get_available_providers()
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(
                "DmlExecutionProvider is not available. "
                "Install/use onnxruntime-directml and a DirectX 12 compatible GPU."
            )

        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.log_severity_level = 3

        self._session = ort.InferenceSession(
            self._onnx_path,
            providers=["DmlExecutionProvider"],
            sess_options=sess_options,
        )
        self._input_name = self._session.get_inputs()[0].name
        self.device_name = "dml"

    def _normalize_wave_batch(self, wave_batch: np.ndarray) -> np.ndarray:
        if wave_batch.ndim != 2:
            raise ValueError(f"Expected [batch, samples], got {wave_batch.shape}")

        batch = np.array(wave_batch, dtype=np.float32, order="C", copy=True)
        cur_samples = batch.shape[1]
        if cur_samples < self.window_samples:
            pad = self.window_samples - cur_samples
            batch = np.pad(batch, ((0, 0), (0, pad)), mode="constant")
        elif cur_samples > self.window_samples:
            batch = batch[:, : self.window_samples]
        return batch

    def _build_mel(self, wave_batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        if torch is None or self._mel is None:
            raise RuntimeError("EfficientAT mel preprocessor is not initialized.")

        with torch.no_grad():
            waveform = torch.from_numpy(wave_batch)
            melspec = self._mel(waveform)
            model_input = melspec.unsqueeze(1).detach().cpu().numpy()
        return np.asarray(model_input, dtype=np.float32)

    def predict_batch(self, wave_batch: np.ndarray) -> np.ndarray:
        if not self.ready or self._session is None or not self._input_name:
            raise RuntimeError(f"EfficientAT backend not ready: {self.load_error}")

        batch = self._normalize_wave_batch(wave_batch)
        model_input = self._build_mel(batch)

        outputs = []
        for start in range(0, model_input.shape[0], self.onnx_batch_size):
            chunk = model_input[start : start + self.onnx_batch_size]
            chunk_len = chunk.shape[0]

            padded = np.zeros(
                (self.onnx_batch_size, 1, self.n_mels, 1000),
                dtype=np.float32,
            )
            padded[:chunk_len] = chunk

            out = self._session.run(None, {self._input_name: padded})[0]
            outputs.append(np.asarray(out[:chunk_len], dtype=np.float32))

        if not outputs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
