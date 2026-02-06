import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from .efficientat_backend import EfficientATBackend
from .utils import get_ffmpeg_path

SAMPLE_RATE = 32000

# Inference pacing
GPU_THROTTLE_SEC_LOW = 0.0060
GPU_THROTTLE_SEC_HIGH = 0.0150

# Windowing
WINDOW_SECONDS = 10.0
HOP_SECONDS = 2.5

# High-accuracy model (single path)
EFFICIENTAT_MODEL_NAME = "mn10_as"
EFFICIENTAT_ONNX_BATCH = 32

# Scoring weights (laughter priority)
EVENT_TARGET_WEIGHT = 0.30
EVENT_PEAK_WEIGHT = 0.10
LAUGHTER_PRIORITY_WEIGHT = 0.55
RMS_WEIGHT = 0.05
LAUGHTER_BOOST_THRESHOLD = 0.35
LAUGHTER_BOOST_GAIN = 0.25
LAUGHTER_BOOST_CAP = 0.20

# Post-processing
CONTINUITY_TRIGGER = 0.35
CONTINUITY_BONUS = 0.08
SMOOTHING_KERNEL = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
SCORE_QUANTILE = 0.82
MIN_ABSOLUTE_SCORE = 0.08
HYSTERESIS_GAP = 0.06
GROUP_GAP_SECONDS = 4.0

# Candidate shaping
MIN_CANDIDATES_MEDIUM = 6     # >= 15 min
MIN_CANDIDATES_LONG = 10      # >= 30 min
MIN_CANDIDATES_VERY_LONG = 14  # >= 90 min
EXTRA_PEAK_MIN_DISTANCE_SEC = 30.0
EXTRA_PEAK_FLOOR = 0.03
EXTRA_PEAK_LAUGHTER_FLOOR = 0.08
EXTRA_PEAK_NONLAUGH_SCORE_MULT = 1.35
EXTRA_CANDIDATE_PENALTY = 0.92
EXTRA_CANDIDATE_LOW_LAUGHTER_PENALTY = 0.88
EXTRA_CANDIDATE_LOW_LAUGHTER_THRESHOLD = 0.12

# Display score normalization
DISPLAY_SCORE_BASE_QUANTILE = 0.60
DISPLAY_SCORE_TOP_QUANTILE = 0.98
DISPLAY_SCORE_RAW_WEIGHT = 0.80

# Group score fusion
GROUP_PEAK_WEIGHT = 0.40
GROUP_P90_WEIGHT = 0.25
GROUP_MEAN_WEIGHT = 0.15
GROUP_LAUGHTER_PEAK_WEIGHT = 0.20
GROUP_LAUGHTER_BOOST_THRESHOLD = 0.35
GROUP_LAUGHTER_BOOST_GAIN = 0.20
GROUP_LAUGHTER_BOOST_CAP = 0.12


@dataclass
class HighlightCandidate:
    start: float
    end: float
    score: float
    details: Optional[dict] = None

    def duration(self):
        return self.end - self.start


class Analyzer:
    def __init__(self):
        self._target_indices: List[int] = []
        self._laughter_indices: List[int] = []
        self._class_names: List[str] = []
        self._labels_ready = False

        self._efficientat: Optional[EfficientATBackend] = None
        self._backend_name = "efficientat_onnx"

    @staticmethod
    def _is_gpu_backend_name(backend_name: str) -> bool:
        return any(tag in backend_name for tag in ("_dml", "_cuda", "_gpu"))

    def _build_target_indices(self):
        targets_laughter = ["Laughter", "Belly laugh", "Chuckle, chortle", "Giggle", "Snicker"]
        targets_other = ["Cheering", "Applause", "Clapping", "Crowd", "Battle cry", "Screaming", "Shouting", "Yell"]

        self._target_indices = []
        self._laughter_indices = []
        for i, name in enumerate(self._class_names):
            is_laughter = any(t.lower() in name.lower() for t in targets_laughter)
            is_other = any(t.lower() in name.lower() for t in targets_other)
            if is_laughter:
                self._laughter_indices.append(i)
                self._target_indices.append(i)
            elif is_other:
                self._target_indices.append(i)

    def _load_label_metadata(self):
        if self._labels_ready:
            return

        if not self._efficientat or not self._efficientat.ready or not self._efficientat.labels:
            raise RuntimeError("EfficientAT labels are unavailable.")

        self._class_names = list(self._efficientat.labels)
        self._build_target_indices()
        self._labels_ready = True
        print(
            f"Target Indices: {len(self._target_indices)} found "
            f"({len(self._laughter_indices)} are laughter)."
        )

    def _load_efficientat_model(self, status_cb: Optional[Callable[[str], None]] = None):
        if self._efficientat and self._efficientat.ready:
            self._backend_name = f"efficientat_onnx_{self._efficientat.device_name}"
            return

        self._efficientat = EfficientATBackend(
            model_name=EFFICIENTAT_MODEL_NAME,
            onnx_batch_size=EFFICIENTAT_ONNX_BATCH,
        )
        if not self._efficientat.ready:
            msg = f"EfficientAT ONNX initialization failed: {self._efficientat.load_error}"
            if status_cb:
                status_cb(msg)
            raise RuntimeError(msg)

        self._backend_name = f"efficientat_onnx_{self._efficientat.device_name}"
        if not self._labels_ready and self._efficientat.labels:
            self._class_names = list(self._efficientat.labels)
            self._build_target_indices()
            self._labels_ready = True
        print(f"EfficientAT ONNX backend ready: {self._backend_name}")

    def _load_model(self, status_cb: Optional[Callable[[str], None]] = None):
        self._load_efficientat_model(status_cb=status_cb)
        self._load_label_metadata()

    def _score_from_probs(
        self,
        clipwise_output: np.ndarray,
        batch_audio: np.ndarray,
        target_indices: np.ndarray,
        laughter_indices: np.ndarray,
        rms_low: float,
        rms_high: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        current_batch_len = clipwise_output.shape[0]
        has_targets = target_indices.size > 0
        has_laughter = laughter_indices.size > 0

        if has_targets:
            valid_probs = clipwise_output[:, target_indices]
            top_peak = np.max(valid_probs, axis=1)
            top_k = min(3, valid_probs.shape[1])
            top_k_idx = np.argpartition(valid_probs, -top_k, axis=1)[:, -top_k:]
            top_k_mean = np.take_along_axis(valid_probs, top_k_idx, axis=1).mean(axis=1)
            score_all = (0.6 * top_peak) + (0.4 * top_k_mean)

            local_max_idx = np.argmax(valid_probs, axis=1)
            top_class_scores = top_peak
            global_idx = target_indices[local_max_idx]
        else:
            score_all = np.zeros(current_batch_len, dtype=np.float32)
            top_class_scores = np.zeros(current_batch_len, dtype=np.float32)
            global_idx = None

        if has_laughter:
            laugh_probs = clipwise_output[:, laughter_indices]
            laugh_peak = np.max(laugh_probs, axis=1)
            laugh_k = min(2, laugh_probs.shape[1])
            laugh_idx = np.argpartition(laugh_probs, -laugh_k, axis=1)[:, -laugh_k:]
            laugh_k_mean = np.take_along_axis(laugh_probs, laugh_idx, axis=1).mean(axis=1)
            score_laughter = (0.7 * laugh_peak) + (0.3 * laugh_k_mean)
        else:
            score_laughter = np.zeros(current_batch_len, dtype=np.float32)

        event_score = (
            (EVENT_TARGET_WEIGHT * score_all)
            + (EVENT_PEAK_WEIGHT * top_class_scores)
            + (LAUGHTER_PRIORITY_WEIGHT * score_laughter)
        )
        laugh_boost = np.clip(
            (score_laughter - LAUGHTER_BOOST_THRESHOLD) * LAUGHTER_BOOST_GAIN,
            0.0,
            LAUGHTER_BOOST_CAP,
        )
        event_score = np.clip(event_score + laugh_boost, 0.0, 1.0)

        batch_rms = np.sqrt(np.mean(batch_audio ** 2, axis=1))
        rms_span = max(1e-6, rms_high - rms_low)
        rms_score = np.clip((batch_rms - rms_low) / rms_span, 0.0, 1.0)
        combined_score = np.clip(event_score + (RMS_WEIGHT * rms_score), 0.0, 1.0)

        has_laughter_flags = score_laughter > 0.15
        if has_targets:
            details = [
                {
                    "top_class": self._class_names[int(global_idx[k])],
                    "max_score": float(top_class_scores[k]),
                    "score_event": float(event_score[k]),
                    "score_laughter": float(score_laughter[k]),
                    "has_laughter": bool(has_laughter_flags[k]),
                }
                for k in range(current_batch_len)
            ]
        else:
            details = [
                {
                    "top_class": "Unknown",
                    "max_score": 0.0,
                    "score_event": float(event_score[k]),
                    "score_laughter": float(score_laughter[k]),
                    "has_laughter": bool(has_laughter_flags[k]),
                }
                for k in range(current_batch_len)
            ]

        return combined_score.astype(np.float32), score_laughter.astype(np.float32), details

    def _compute_rms_calibration(self, wav_data: np.ndarray) -> Tuple[float, float]:
        if wav_data.size == 0:
            return 0.0, 1.0

        rms_frame = max(256, int(0.10 * SAMPLE_RATE))
        rms_hop = max(128, int(0.05 * SAMPLE_RATE))
        frame_len = min(rms_frame, wav_data.size)
        hop_len = min(rms_hop, max(1, wav_data.size))

        rms_track = librosa.feature.rms(
            y=wav_data,
            frame_length=frame_len,
            hop_length=hop_len,
        )[0]
        if rms_track.size == 0:
            level = float(np.sqrt(np.mean(wav_data ** 2)))
            return max(0.0, level * 0.5), max(1e-4, level * 1.5)

        rms_low = float(np.quantile(rms_track, 0.20))
        rms_high = float(np.quantile(rms_track, 0.90))
        if rms_high <= rms_low + 1e-6:
            peak = float(np.max(rms_track))
            rms_high = max(rms_low + 1e-4, peak + 1e-6)
        return rms_low, rms_high

    def _compute_group_candidate_score(
        self,
        final_scores: np.ndarray,
        laughter_scores: np.ndarray,
        group_indices: np.ndarray,
        is_extra: bool,
    ) -> Tuple[float, dict]:
        group_score_values = final_scores[group_indices]
        group_laughter_values = laughter_scores[group_indices] if laughter_scores.size > 0 else np.zeros_like(group_score_values)

        group_peak = float(np.max(group_score_values))
        group_mean = float(np.mean(group_score_values))
        group_p90 = float(np.quantile(group_score_values, 0.90))
        group_laughter_peak = float(np.max(group_laughter_values)) if group_laughter_values.size > 0 else 0.0

        score = (
            (GROUP_PEAK_WEIGHT * group_peak)
            + (GROUP_P90_WEIGHT * group_p90)
            + (GROUP_MEAN_WEIGHT * group_mean)
            + (GROUP_LAUGHTER_PEAK_WEIGHT * group_laughter_peak)
        )
        laughter_boost = np.clip(
            (group_laughter_peak - GROUP_LAUGHTER_BOOST_THRESHOLD) * GROUP_LAUGHTER_BOOST_GAIN,
            0.0,
            GROUP_LAUGHTER_BOOST_CAP,
        )
        score = float(np.clip(score + laughter_boost, 0.0, 1.0))

        if is_extra:
            score *= EXTRA_CANDIDATE_PENALTY
            if group_laughter_peak < EXTRA_CANDIDATE_LOW_LAUGHTER_THRESHOLD:
                score *= EXTRA_CANDIDATE_LOW_LAUGHTER_PENALTY
            score = float(np.clip(score, 0.0, 1.0))

        score_breakdown = {
            "score_group_peak": group_peak,
            "score_group_p90": group_p90,
            "score_group_mean": group_mean,
            "score_group_laughter_peak": group_laughter_peak,
            "is_extra_candidate": bool(is_extra),
        }
        return score, score_breakdown

    def _build_active_mask(self, scores: np.ndarray) -> Tuple[np.ndarray, float, float]:
        if scores.size == 0:
            return np.zeros(0, dtype=bool), 0.0, 0.0

        threshold_quantile = float(np.quantile(scores, SCORE_QUANTILE))
        threshold_mean = float(np.mean(scores) + 0.5 * np.std(scores))
        threshold_start = max(MIN_ABSOLUTE_SCORE, threshold_quantile, threshold_mean)
        threshold_end = max(MIN_ABSOLUTE_SCORE * 0.75, threshold_start - HYSTERESIS_GAP)

        active = scores >= threshold_start
        if active.size > 2:
            for i in range(1, active.size - 1):
                if active[i]:
                    continue
                if scores[i] >= threshold_end and (active[i - 1] or active[i + 1]):
                    active[i] = True

        # If nothing passed, keep the strongest peak when it is at least mildly salient.
        if not np.any(active):
            peak_idx = int(np.argmax(scores))
            peak_score = float(scores[peak_idx])
            soft_floor = max(0.10, float(np.mean(scores) + 0.25 * np.std(scores)))
            if peak_score >= soft_floor:
                active[peak_idx] = True
                neighbor_floor = soft_floor * 0.90
                if peak_idx > 0 and scores[peak_idx - 1] >= neighbor_floor:
                    active[peak_idx - 1] = True
                if peak_idx + 1 < scores.size and scores[peak_idx + 1] >= neighbor_floor:
                    active[peak_idx + 1] = True

        return active, threshold_start, threshold_end

    def _group_active(self, active_indices: np.ndarray, times: np.ndarray) -> List[Tuple[int, int]]:
        groups: List[Tuple[int, int]] = []
        if active_indices.size == 0:
            return groups

        g_start = int(active_indices[0])
        g_prev = g_start
        for idx in active_indices[1:]:
            idx_int = int(idx)
            if times[idx_int] - times[g_prev] <= GROUP_GAP_SECONDS:
                g_prev = idx_int
            else:
                groups.append((g_start, g_prev))
                g_start = idx_int
                g_prev = idx_int
        groups.append((g_start, g_prev))
        return groups

    def _target_candidate_count(self, duration_sec: float) -> int:
        if duration_sec >= 5400.0:
            return MIN_CANDIDATES_VERY_LONG
        if duration_sec >= 1800.0:
            return MIN_CANDIDATES_LONG
        if duration_sec >= 900.0:
            return MIN_CANDIDATES_MEDIUM
        return 1

    def _pick_extra_peak_indices(
        self,
        scores: np.ndarray,
        times: np.ndarray,
        existing_indices: List[int],
        need_count: int,
        laughter_scores: Optional[np.ndarray] = None,
    ) -> List[int]:
        if need_count <= 0 or scores.size == 0:
            return []

        sorted_idx = np.argsort(scores)[::-1]
        picked: List[int] = []
        blocked = set(int(i) for i in existing_indices)

        soft_floor = max(
            EXTRA_PEAK_FLOOR,
            float(np.quantile(scores, 0.60) * 0.65),
        )

        for idx in sorted_idx:
            idx_int = int(idx)
            if idx_int in blocked:
                continue
            if float(scores[idx_int]) < soft_floor:
                break
            if laughter_scores is not None:
                laugh_val = float(laughter_scores[idx_int])
                score_val = float(scores[idx_int])
                if laugh_val < EXTRA_PEAK_LAUGHTER_FLOOR and score_val < (soft_floor * EXTRA_PEAK_NONLAUGH_SCORE_MULT):
                    continue

            too_close = False
            for used_idx in existing_indices:
                if abs(float(times[idx_int]) - float(times[used_idx])) < EXTRA_PEAK_MIN_DISTANCE_SEC:
                    too_close = True
                    break
            if too_close:
                continue
            for used_idx in picked:
                if abs(float(times[idx_int]) - float(times[used_idx])) < EXTRA_PEAK_MIN_DISTANCE_SEC:
                    too_close = True
                    break
            if too_close:
                continue

            picked.append(idx_int)
            if len(picked) >= need_count:
                break

        # If still short, relax score floor and fill by distance-constrained peaks.
        if len(picked) < need_count:
            for idx in sorted_idx:
                idx_int = int(idx)
                if idx_int in blocked or idx_int in picked:
                    continue

                too_close = False
                for used_idx in existing_indices:
                    if abs(float(times[idx_int]) - float(times[used_idx])) < EXTRA_PEAK_MIN_DISTANCE_SEC:
                        too_close = True
                        break
                if too_close:
                    continue
                for used_idx in picked:
                    if abs(float(times[idx_int]) - float(times[used_idx])) < EXTRA_PEAK_MIN_DISTANCE_SEC:
                        too_close = True
                        break
                if too_close:
                    continue

                picked.append(idx_int)
                if len(picked) >= need_count:
                    break

        return picked

    def extract_audio(self, video_path: str, output_wav_path: str):
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found.")

        command = [
            ffmpeg_path,
            "-y",
            "-threads",
            "2",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            output_wav_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def analyze_audio(
        self,
        wav_path: str,
        padding: float = 20.0,
        progress_cb: Optional[Callable[[float], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        batch_size: int = 32,
    ) -> List[HighlightCandidate]:
        batch_size = max(1, int(batch_size))
        self._load_model(status_cb=status_cb)

        if status_cb:
            status_cb("Loading audio...")
        print(f"Loading audio {wav_path}...")

        wav_data, sr = sf.read(wav_path, dtype="float32")
        if wav_data.ndim > 1:
            wav_data = np.mean(wav_data, axis=1)
        if sr != SAMPLE_RATE:
            wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav_data = np.ascontiguousarray(wav_data, dtype=np.float32)

        if wav_data.size == 0:
            print("Warning: Empty audio data.")
            return []

        total_samples = len(wav_data)
        window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        hop_samples = int(HOP_SECONDS * SAMPLE_RATE)

        if len(wav_data) < window_samples:
            wav_data = np.pad(wav_data, (0, window_samples - len(wav_data)), mode="constant")

        pad_mod = (len(wav_data) - window_samples) % hop_samples
        if pad_mod != 0:
            wav_data = np.pad(wav_data, (0, hop_samples - pad_mod), mode="constant")

        try:
            frames = librosa.util.frame(
                wav_data,
                frame_length=window_samples,
                hop_length=hop_samples,
            ).T
        except Exception as e:
            print(f"Frame creation failed: {e}")
            if status_cb:
                status_cb("Audio frame creation failed.")
            return []

        num_windows = len(frames)
        if num_windows == 0:
            return []

        target_indices = np.asarray(self._target_indices, dtype=np.int64)
        laughter_indices = np.asarray(self._laughter_indices, dtype=np.int64)

        times = (np.arange(num_windows) * hop_samples + (window_samples / 2.0)) / SAMPLE_RATE
        window_details: List[Optional[dict]] = [None] * num_windows
        raw_scores_arr = np.empty(num_windows, dtype=np.float32)
        laughter_scores_arr = np.zeros(num_windows, dtype=np.float32)

        rms_low, rms_high = self._compute_rms_calibration(wav_data)
        print(f"RMS calibration: low={rms_low:.5f}, high={rms_high:.5f}")

        if progress_cb:
            progress_cb(0.1)
        if status_cb:
            status_cb(f"Inference running ({self._backend_name} / batch={batch_size})...")
        print(f"Inference backend: {self._backend_name}, windows={num_windows}, batch={batch_size}")

        is_gpu_backend = self._is_gpu_backend_name(self._backend_name)
        gpu_throttle_sec = (
            GPU_THROTTLE_SEC_HIGH if (is_gpu_backend and batch_size >= 32) else GPU_THROTTLE_SEC_LOW
        )
        if not is_gpu_backend:
            gpu_throttle_sec = 0.0

        if not self._efficientat or not self._efficientat.ready:
            raise RuntimeError("EfficientAT backend is not initialized.")

        for i in range(0, num_windows, batch_size):
            batch_end = min(i + batch_size, num_windows)
            batch_windows = np.ascontiguousarray(frames[i:batch_end], dtype=np.float32)

            clipwise_output = self._efficientat.predict_batch(batch_windows)

            combined_score, score_laughter, details = self._score_from_probs(
                clipwise_output=clipwise_output,
                batch_audio=batch_windows,
                target_indices=target_indices,
                laughter_indices=laughter_indices,
                rms_low=rms_low,
                rms_high=rms_high,
            )
            raw_scores_arr[i:batch_end] = combined_score
            laughter_scores_arr[i:batch_end] = score_laughter
            window_details[i:batch_end] = details

            if progress_cb:
                percent = 0.15 + (0.75 * batch_end / num_windows)
                progress_cb(min(0.90, percent))

            if gpu_throttle_sec > 0.0:
                time.sleep(gpu_throttle_sec)

        final_scores = raw_scores_arr.copy()
        if final_scores.size > 1:
            high = final_scores >= CONTINUITY_TRIGGER
            final_scores[1:] += high[:-1] * CONTINUITY_BONUS
            final_scores[:-1] += high[1:] * CONTINUITY_BONUS
        final_scores = np.clip(final_scores, 0.0, 1.0)

        if final_scores.size >= 3:
            final_scores = np.convolve(final_scores, SMOOTHING_KERNEL, mode="same")
            final_scores = np.clip(final_scores, 0.0, 1.0)

        active_mask, threshold_start, threshold_end = self._build_active_mask(final_scores)
        active_indices = np.where(active_mask)[0]

        print(
            f"Scoring thresholds: start={threshold_start:.3f}, end={threshold_end:.3f}, "
            f"active={len(active_indices)}"
        )
        if progress_cb:
            progress_cb(0.95)

        groups = self._group_active(active_indices, times)
        if not groups and final_scores.size > 0:
            peak_idx = int(np.argmax(final_scores))
            groups = [(peak_idx, peak_idx)]
            print("No windows passed threshold; using peak fallback candidate.")
        duration = total_samples / float(SAMPLE_RATE)
        target_candidates = self._target_candidate_count(duration)

        peak_indices: List[int] = []
        for g_start_idx, g_end_idx in groups:
            group_indices = np.arange(g_start_idx, g_end_idx + 1)
            peak_idx = int(group_indices[np.argmax(final_scores[group_indices])])
            peak_indices.append(peak_idx)

        extras: List[int] = []
        if len(peak_indices) < target_candidates:
            extras = self._pick_extra_peak_indices(
                scores=final_scores,
                times=times,
                existing_indices=peak_indices,
                need_count=target_candidates - len(peak_indices),
                laughter_scores=laughter_scores_arr,
            )
            if extras:
                print(f"Adding {len(extras)} extra peak candidates.")
                for idx in extras:
                    groups.append((idx, idx))
                    peak_indices.append(idx)
        extra_index_set = set(int(x) for x in extras)

        global_score_base = float(np.quantile(final_scores, DISPLAY_SCORE_BASE_QUANTILE))
        global_score_top = float(np.quantile(final_scores, DISPLAY_SCORE_TOP_QUANTILE))
        global_score_span = max(1e-6, global_score_top - global_score_base)

        candidates = []
        pre_padding = min(8.0, max(3.0, padding * 0.25))
        post_padding = max(10.0, padding)

        half_window = WINDOW_SECONDS / 2.0
        for g_start_idx, g_end_idx in groups:
            group_indices = np.arange(g_start_idx, g_end_idx + 1)
            peak_idx = int(group_indices[np.argmax(final_scores[group_indices])])
            is_extra = peak_idx in extra_index_set

            group_score, score_breakdown = self._compute_group_candidate_score(
                final_scores=final_scores,
                laughter_scores=laughter_scores_arr,
                group_indices=group_indices,
                is_extra=is_extra,
            )
            rel_global = float(np.clip((group_score - global_score_base) / global_score_span, 0.0, 1.0))
            score = float(
                np.clip(
                    (DISPLAY_SCORE_RAW_WEIGHT * group_score)
                    + ((1.0 - DISPLAY_SCORE_RAW_WEIGHT) * rel_global),
                    0.0,
                    1.0,
                )
            )

            center_start = float(times[g_start_idx])
            center_end = float(times[g_end_idx])
            s = max(0.0, center_start - half_window - pre_padding)
            e = min(duration, center_end + half_window + post_padding)

            det = window_details[peak_idx] if peak_idx < len(window_details) else None
            if det is None:
                det = {}
            else:
                det = dict(det)
            det.update(score_breakdown)
            candidates.append(
                HighlightCandidate(
                    start=round(s, 2),
                    end=round(e, 2),
                    score=round(score, 4),
                    details=det,
                )
            )

        candidates.sort(key=lambda x: x.start)
        if progress_cb:
            progress_cb(1.0)
        return candidates


analyzer = Analyzer()
