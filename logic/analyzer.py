import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
import csv
import time
from dataclasses import dataclass
from typing import List, Optional, Callable
from .utils import get_ffmpeg_path, get_resource_path

# PANNs Constants
MODEL_FILENAME = "Cnn14.onnx"
CLASSES_FILENAME = "panns_classes.txt"
SAMPLE_RATE = 32000

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
        self._session = None
        self._target_indices = []
        self._class_names = []

    def _load_model(self, batch_size: int = 32):
        # Dynamic model name based on batch size
        target_model_name = f"Cnn14_batch{batch_size}.onnx"
        # Use get_resource_path for bundled models
        model_path = get_resource_path(target_model_name)
        
        if not os.path.exists(model_path):
             # Run setup for this specific batch size
             print(f"Model {target_model_name} not found at {model_path}.")
             # In bundled mode, we cannot run setup_model as it requires torch
             # So we just warn/error or fallback if possible
             pass
             
        # Apply session options
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.log_severity_level = 3 # Error only
        
        # Check providers
        available_providers = ort.get_available_providers()
        providers = []
        if 'DmlExecutionProvider' in available_providers:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')

        if self._session is None:
             # Just create session
             try:
                self._session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
                self._loaded_batch_size = batch_size
             except Exception as e:
                 print(f"Failed to init DirectML session: {e}")
                 # Fallback cpu
                 self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)
                 self._loaded_batch_size = batch_size
        else:
            # Reload if batch size changed
            if not hasattr(self, '_loaded_batch_size') or self._loaded_batch_size != batch_size:
                 print(f"Reloading session for new batch size {batch_size}")
                 try:
                    self._session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
                    self._loaded_batch_size = batch_size
                 except Exception as e:
                     print(f"Failed to init DirectML session: {e}")
                     self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=sess_options)
                     self._loaded_batch_size = batch_size
        
        # Load Classes
        classes_path = get_resource_path(CLASSES_FILENAME)
        if os.path.exists(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                self._class_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback/Error
            raise FileNotFoundError("panns_classes.txt not found")
        
        # Define Targets (PANNs labels are specific)
        targets = [
            "Laughter", "Belly laugh", "Chuckle, chortle", "Giggle", "Snicker", 
            "Cheering", "Applause", "Clapping", "Crowd", "Battle cry", 
            "Screaming", "Shouting", "Yell"
        ]
        
        self._target_indices = []
        for i, name in enumerate(self._class_names):
            # Case insensitive check
            if any(t.lower() in name.lower() for t in targets):
                self._target_indices.append(i)
        
        print(f"Target Indices: {len(self._target_indices)} found.")

    def extract_audio(self, video_path: str, output_wav_path: str):
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found.")

        # Resample to 32000 for PANNs
        command = [
            ffmpeg_path,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            output_wav_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def analyze_audio(self, wav_path: str, padding: float = 20.0, progress_cb: Optional[Callable[[float], None]] = None, status_cb: Optional[Callable[[str], None]] = None, batch_size: int = 32) -> List[HighlightCandidate]:
        self._load_model(batch_size=batch_size)
        
        if status_cb: status_cb("音声データを読み込み中...")
        print(f"Loading audio {wav_path}...")
        # Use soundfile for faster reading (requires 32000Hz mono wav already)
        wav_data, sr = sf.read(wav_path, dtype='float32')
        # wav_data, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        
        total_samples = len(wav_data)
        
        # Sliding Window Config
        window_seconds = 10.0
        window_samples = int(window_seconds * SAMPLE_RATE)
        hop_seconds = 5.0 # Increased from 2.0 for 2.5x speedup
        hop_samples = int(hop_seconds * SAMPLE_RATE)
        
        num_windows = int(np.ceil((total_samples - window_samples) / hop_samples)) + 1
        if num_windows < 1: num_windows = 1
        
        # Define Targets (PANNs labels are specific)
        targets_laughter = ["Laughter", "Belly laugh", "Chuckle, chortle", "Giggle", "Snicker"]
        targets_other = ["Cheering", "Applause", "Clapping", "Crowd", "Battle cry", "Screaming", "Shouting", "Yell"]
        
        self._target_indices = []
        self._laughter_indices = []
        window_details = []
        
        for i, name in enumerate(self._class_names):
            # Case insensitive check
            is_laughter = any(t.lower() in name.lower() for t in targets_laughter)
            is_other = any(t.lower() in name.lower() for t in targets_other)
            
            if is_laughter:
                self._laughter_indices.append(i)
                self._target_indices.append(i)
            elif is_other:
                self._target_indices.append(i)
        
        print(f"Target Indices: {len(self._target_indices)} found ({len(self._laughter_indices)} are laughter).")
        
        all_scores_time_map = [] 
        
        if progress_cb: progress_cb(0.1)
        
        # Check current provider
        current_providers = self._session.get_providers()
        provider_msg = "GPU" if 'DmlExecutionProvider' in current_providers else "CPU"
        if status_cb: status_cb(f"推論実行中 ({provider_msg} / Batch={batch_size})...")
        print(f"Inference started. Providers: {current_providers}, Batch Size: {batch_size}")

        # Batch inference is now dynamic based on argument
        
        if status_cb: status_cb("音声データを前処理中 (一括変換)...")
        print(f"Pre-processing audio windows...")
        
        # Handle cases where audio is empty or extremely short
        if len(wav_data) == 0:
            print("Warning: Empty audio data.")
            return []
            
        # Ensure minimum length for at least one window
        min_samples = window_samples
        if len(wav_data) < min_samples:
            print(f"Audio too short ({len(wav_data)} samples), padding to {min_samples}...")
            wav_data = np.pad(wav_data, (0, min_samples - len(wav_data)), mode='constant')

        # Vectorized Sliding Window using librosa.util.frame
        # Pad audio to ensure integer number of frames
        # (total - window) % hop should be 0 for perfect fit with center=False behavior of frame?
        # Actually librosa frame just takes snapshots.
        # We want to maintain coverage.
        
        # Safe padding
        pad_length = window_samples - (len(wav_data) - window_samples) % hop_samples
        if pad_length < window_samples:
             wav_data_padded = np.pad(wav_data, (0, pad_length), mode='constant')
        else:
             wav_data_padded = wav_data
        
        # Extra safety check for very short padded data (shouldn't happen with logic above)
        if len(wav_data_padded) < window_samples:
             wav_data_padded = np.pad(wav_data_padded, (0, window_samples - len(wav_data_padded)), mode='constant')

        try:
            frames = librosa.util.frame(wav_data_padded, frame_length=window_samples, hop_length=hop_samples).T.copy()
        except Exception as e:
            print(f"Frame creation failed: {e}")
            if status_cb: status_cb("音声処理エラー (Frame)")
            return []
        
        num_windows = len(frames)
        times = [(i * hop_samples + window_samples/2) / SAMPLE_RATE for i in range(num_windows)]
        
        print(f"Generated {num_windows} windows. Starting batch inference...")
        
        if progress_cb: progress_cb(0.15)
        
        # Check current provider
        current_providers = self._session.get_providers()
        provider_msg = "GPU" if 'DmlExecutionProvider' in current_providers else "CPU"
        if status_cb: status_cb(f"推論実行中 ({provider_msg} / Batch={batch_size})...")
        
        raw_scores = []
        input_name = self._session.get_inputs()[0].name
        
        # Batch Loop
        for i in range(0, num_windows, batch_size):
            # Slice batch
            batch_windows = frames[i : i + batch_size]
            current_batch_len = len(batch_windows)
            
            # Pad batch if needed (for DirectML stability)
            if current_batch_len < batch_size:
                padding = np.zeros((batch_size - current_batch_len, window_samples), dtype=np.float32)
                batch_input = np.vstack([batch_windows, padding])
            else:
                batch_input = batch_windows
            
            batch_input = batch_input.astype(np.float32)
            
            # Silence Optimization
            # Check max amplitude in batch
            max_amp = np.max(np.abs(batch_input))
            # Threshold: 0.005 is roughly -46dB. If max is below this, it's very quiet.
            if max_amp < 0.005:
                 # Skip inference, assume zero probability
                 # Output shape: [batch_size, 527]
                 # We need to construct dummy output
                 dummy_output = np.zeros((current_batch_len, 527), dtype=np.float32)
                 outputs = [dummy_output] # List wrapping to match session.run result
                 # print(f"Skipping silent batch {i}")
            else:
                try:
                    outputs = self._session.run(None, {input_name: batch_input})
                except Exception as e:
                    err_msg = str(e)
                    print(f"Inference error: {err_msg}")
                    # Fallback logic if needed, but let's hope pre-check worked.
                    # If Dml crashes here, we might need full fallback.
                    if 'DmlExecutionProvider' in self._session.get_providers():
                         print("Runtime Fallback to CPU...")
                         self._session = ort.InferenceSession(os.path.join(os.getcwd(), MODEL_FILENAME), providers=['CPUExecutionProvider'])
                         outputs = self._session.run(None, {input_name: batch_input})
                    else:
                        raise e

            # Slice output
            clipwise_output = outputs[0][:current_batch_len]
            
            # Process Scores (Vectorized where possible? For now loop is fine as it's small 32 items)
            # Actually, standardizing this part:
            for j in range(current_batch_len):
                prob_vec = clipwise_output[j]
                
                # Targets
                score_laughter = 0.0
                if self._laughter_indices:
                    score_laughter = np.max(prob_vec[self._laughter_indices])
                score_all = np.max(prob_vec[self._target_indices])
                
                clip_score = score_all
                if score_laughter > 0.1:
                    clip_score = max(clip_score, score_laughter * 1.2)
                
                clip_score = np.clip(clip_score * 1.5, 0.0, 1.0)
                
                # RMS (already loaded in memory)
                current_chunk = batch_windows[j]
                rms = np.sqrt(np.mean(current_chunk**2))
                rms_score = min(1.0, rms / 0.1)
                
                combined_score = 0.6 * clip_score + 0.4 * rms_score
                raw_scores.append(combined_score)
                
                # Capture Details
                top_class = "Unknown"
                top_class_score = 0.0
                
                # Find max in target indices
                valid_probs = prob_vec[self._target_indices]
                if len(valid_probs) > 0:
                    local_max_idx = np.argmax(valid_probs)
                    global_idx = self._target_indices[local_max_idx]
                    top_class = self._class_names[global_idx]
                    top_class_score = float(valid_probs[local_max_idx])
                
                window_details.append({
                    "top_class": top_class,
                    "max_score": top_class_score,
                    "has_laughter": (score_laughter > 0.1)
                })
            
            # Progress update
            if progress_cb:
                percent = 0.15 + (0.8 * (i + batch_size) / num_windows)
                progress_cb(min(0.95, percent))
            
            # Yield to system (UI/Video Playback) to prevent lag
            time.sleep(0.02)

        if not raw_scores:
             return []
             
        # 4. Continuity Boost (Post-processing)
        # Add bonus if adjacent segments are also high
        final_scores = []
        count = len(raw_scores)
        continuity_bonus = 0.15
        
        for i in range(count):
            score = raw_scores[i]
            
            # Check previous
            if i > 0 and raw_scores[i-1] > 0.3:
                score += continuity_bonus
                
            # Check next
            if i < count - 1 and raw_scores[i+1] > 0.3:
                score += continuity_bonus
            
            final_scores.append(min(1.0, score))
            
        # Re-pack into time map for downstream logic (replacing raw with final)
        # Note: 'times' list contains the center time for each window processed in order.
        # 'raw_scores' and 'final_scores' are aligned with 'times'.
        all_scores_time_map = [(times[i], final_scores[i]) for i in range(count)]
        
        times_arr = np.array([x[0] for x in all_scores_time_map])
        scores_arr = np.array([x[1] for x in all_scores_time_map])
        
        # Sort by time (this might be redundant if times are already sorted, but good for safety)
        sorted_idx = np.argsort(times_arr)
        times_arr = times_arr[sorted_idx]
        scores_arr = scores_arr[sorted_idx]
        
        # Quick RMS (downsampled) to mix in volume
        hop_rms = int(SAMPLE_RATE * 0.1) # 0.1s
        rms_frames = librosa.feature.rms(y=wav_data, frame_length=2048, hop_length=hop_rms)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms_frames)), sr=SAMPLE_RATE, hop_length=hop_rms)
        
        # Interpolate PANNs scores to RMS timeline
        scores_interp = np.interp(rms_times, times_arr, scores_arr)
        
        # Normalize RMS
        rms_norm = (rms_frames - np.min(rms_frames)) / (np.max(rms_frames) - np.min(rms_frames) + 1e-6)
        
        # Final Score: 70% PANNs, 30% Volume
        final_scores = (scores_interp * 0.7) + (rms_norm * 0.3)
        
        # Threshold Logic
        threshold = np.mean(final_scores) + 2.0 * np.std(final_scores)
        min_absolute_score = 0.15
        
        active_indices = np.where((final_scores > threshold) & (final_scores > min_absolute_score))[0]
        
        if progress_cb: progress_cb(0.95)
        
        # Group logic (reused)
        groups = []
        if len(active_indices) > 0:
            current_group = [active_indices[0]]
            for i in range(1, len(active_indices)):
                idx = active_indices[i]
                last_idx = active_indices[i-1]
                t_current = rms_times[idx]
                t_last = rms_times[last_idx]
                
                if t_current - t_last < 5.0: # 5s gap
                    current_group.append(idx)
                else:
                    best_idx = max(current_group, key=lambda ix: final_scores[ix])
                    groups.append((current_group[0], current_group[-1], best_idx))
                    current_group = [idx]
            best_idx = max(current_group, key=lambda ix: final_scores[ix])
            groups.append((current_group[0], current_group[-1], best_idx))
            
        candidates = []
        duration = total_samples / float(SAMPLE_RATE)
        
        for g_start, g_end, p_idx in groups:
            t_start = rms_times[g_start]
            t_end = rms_times[g_end]
            score = final_scores[p_idx]
            
            # -5s, +10s Logic
            s = max(0, t_start - 5.0)
            e = min(duration, t_end + 10.0)
            
            # Find nearest original window for details
            # t_peak is roughly the center of the group, or we use t_start/t_end center
            # For simplicity, let's use the peak index time
            t_peak = rms_times[p_idx]
            
            # Find closest index in 'times' (original windows)
            # times is a list/array
            closest_w_idx = 0
            if len(times) > 0:
                closest_w_idx = np.abs(np.array(times) - t_peak).argmin()
            
            det = window_details[closest_w_idx] if closest_w_idx < len(window_details) else None

            candidates.append(HighlightCandidate(
                start=round(float(s), 2),
                end=round(float(min(e, duration)), 2),
                score=round(float(score), 4),
                details=det
            ))
            
        candidates.sort(key=lambda x: x.start)
        if progress_cb: progress_cb(1.0)
        
        return candidates

analyzer = Analyzer()
