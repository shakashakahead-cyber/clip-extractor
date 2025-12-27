import logging
import flet as ft
import os
import threading
import json
from typing import List, Dict
# moviepy import moved to local scope to prevent startup hang
pass

# NOTE: Flet Compatibility Warning
# Flet 0.26+ changes the way colors are handled.
# DO NOT use 'ft.colors.COLOR_NAME' (constants) or 'ft.colors.with_opacity'.
# ALWAYS use string literals for colors (e.g., "black", "blue", "#80FFFFFF").
# "transparent" is also a valid string literal.
# Failure to follow this will cause AttributeError: module 'flet' has no attribute 'colors'.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Single Instance Lock (Windows) ---
import ctypes
import sys

MUTEX_NAME = "ClipExtractorBeta_SingleInstance_Mutex"
_kernel32 = ctypes.windll.kernel32
_mutex = _kernel32.CreateMutexW(None, False, MUTEX_NAME)
_last_error = _kernel32.GetLastError()

ERROR_ALREADY_EXISTS = 183
if _last_error == ERROR_ALREADY_EXISTS:
    # Another instance is running
    ctypes.windll.user32.MessageBoxW(0, "このアプリケーションは既に起動しています。", "クリップ抽出くん ベータ", 0x40)
    sys.exit(0)
# --- End Single Instance Lock ---

print("Script started")

from logic.utils import get_user_data_path, get_resource_path

# Use AppData for config
CONFIG_FILE = get_user_data_path("config.json")

class AppState:
    def __init__(self):
        self.video_path: str = ""
        self.duration: float = 0.0 # Video duration in seconds
        self.candidates = [] 
        self.selected_indices: set = set()
        self.active_index: int = -1 # Currently focused candidate for range editing
        self.min_score: float = 0.3 # Initial Default
        self.batch_size: int = 32 # Default Batch Size
        self.is_playing: bool = False
        self.load_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    self.min_score = data.get("min_score", 0.3)
                    self.batch_size = data.get("batch_size", 16)
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
            except Exception as e:
                logging.error(f"Failed to load config: {e}")
        else:
            # First run or config deleted -> Auto Config
            from logic.gpu_utils import get_recommended_batch_size
            rec_batch = get_recommended_batch_size()
            
            # User requested 32 preference over auto-detect if high-end
            # If rec_batch is 16 but we know it's capable, boost to 32?
            # Actually, let's trust get_recommended_batch_size but default self.batch_size to 32 if detection fails or is conservative
            # OR just overwrite it here since user asked.
            
            # For this specific user request: "Make it 32"
            # Now updated to 64 for high end
            if rec_batch >= 16:
                self.batch_size = rec_batch # Accept 32 or 64
            else:
                self.batch_size = rec_batch

            self.save_config() # Save immediately so user knows
            logging.info(f"Initial Setup: Auto-detected Batch Size = {self.batch_size}")

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({
                    "min_score": self.min_score,
                    "batch_size": self.batch_size
                }, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")

state = AppState()

def main(page: ft.Page):
    print("Entered main function")
    from logic.utils import cleanup_temp_file, format_time, check_ffmpeg
    
    page.title = "クリップ抽出くん ベータ"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.width = 1200
    page.window.height = 900
    page.padding = 20

    # UI Components
    file_picker = ft.FilePicker()
    save_file_picker = ft.FilePicker()
    page.overlay.extend([file_picker, save_file_picker])
    
    status_text = ft.Text("動画ファイルを選択してください。", color="grey700")
    
    progress_bar = ft.ProgressBar(width=400, visible=False, value=0)
    progress_text = ft.Text("", visible=False)

    # Score Slider
    score_label = ft.Text(f"スコアフィルタ: {state.min_score:.2f} 以上", size=14)

    def on_slider_change(e):
        state.min_score = float(e.control.value)
        score_label.value = f"スコアフィルタ: {state.min_score:.2f} 以上"
        score_label.update()
        render_list_items()
    
    def on_slider_change_end(e):
        state.save_config()

    score_slider = ft.Slider(
        min=0.0, max=1.0, divisions=100, 
        value=state.min_score, 
        label="{value}",
        on_change=on_slider_change,
        on_change_end=on_slider_change_end
    )
    
    # Results Header Text
    results_header_text = ft.Text("検出結果", weight=ft.FontWeight.BOLD)

    # --- Timeline Overlay Components ---
    range_label = ft.Text("範囲編集", size=12, color="white", weight=ft.FontWeight.BOLD)
    
    # State for editing
    editing_backup = {}
    sort_mode = None  # Score sort: None (no sort), 'desc' (high to low), 'asc' (low to high)
    edit_mode_active = False  # Flag to track edit mode for position updates
    current_playback_position = 0.0  # Current position in seconds
    playback_start_time = 0.0  # Time when playback started (for position estimation)
    playback_start_position = 0.0  # Position when playback started
    
    # Helper to access video safely
    def get_video_control():
        try:
             # Stack[0] is Video
             return player_stack.controls[0]
        except:
             return None

    def play_video(e):
        v = get_video_control()
        if v and hasattr(v, 'play'): 
            v.play()
            state.is_playing = True

    def pause_video(e):
        v = get_video_control()
        if v and hasattr(v, 'pause'): 
            v.pause()
            state.is_playing = False

    def toggle_playback(e):
        v = get_video_control()
        if v:
            if state.is_playing:
                if hasattr(v, 'pause'): v.pause()
                state.is_playing = False
                show_snack("一時停止", is_error=False) # Visual feedback
            else:
                if hasattr(v, 'play'): v.play()
                state.is_playing = True
                show_snack("再生", is_error=False)
        
    def on_volume_change(e):
        v = get_video_control()
        if v:
            v.volume = float(e.control.value)
            v.update()

    def on_range_change(e):
        if state.active_index < 0 or state.active_index >= len(state.candidates): return
        cand = state.candidates[state.active_index]
        cand.start = float(e.control.start_value)
        cand.end = float(e.control.end_value)
        range_label.value = f"{format_time(cand.start)} - {format_time(cand.end)}"
        range_label.update()
    
    def on_range_change_end(e):
        if state.active_index < 0: return
        # Don't render list items here in Edit Mode (List is hidden)
        # render_list_items() 
        
        cand = state.candidates[state.active_index]
        v = get_video_control()
        if v:
             if hasattr(v, 'seek'): v.seek(int(cand.start * 1000))
             elif hasattr(v, 'jump_to'): v.jump_to(int(cand.start * 1000))

    def on_position_slider_change(e):
        """Seek to position when user drags the position slider"""
        if state.active_index < 0: return
        cand = state.candidates[state.active_index]
        
        # Position slider is constrained to start-end range
        new_pos = float(e.control.value)
        # Clamp to range
        new_pos = max(cand.start, min(cand.end, new_pos))
        
        nonlocal current_playback_position
        current_playback_position = new_pos
        
        v = get_video_control()
        if v:
            if hasattr(v, 'seek'): v.seek(int(new_pos * 1000))
            elif hasattr(v, 'jump_to'): v.jump_to(int(new_pos * 1000))

    def save_edit_mode(e):
        exit_edit_mode(save=True)

    def cancel_edit_mode(e):
        exit_edit_mode(save=False)

    range_slider = ft.RangeSlider(
        min=0, max=100, 
        start_value=0, end_value=0,
        label="{value}",
        on_change=on_range_change,
        on_change_end=on_range_change_end,
        overlay_color="transparent",
        active_color="orange",
        inactive_color="#4DFFFFFF",
        expand=True
    )
    
    # Playback position slider for edit mode
    position_slider = ft.Slider(
        min=0, max=100,
        value=0,
        active_color="cyan",
        inactive_color="#4DFFFFFF",
        thumb_color="white",
        expand=True,
        on_change_end=on_position_slider_change
    )
    
    # Current position label
    position_label = ft.Text("", size=11, color="cyan")
    # Position update timer reference
    position_timer_running = False
    
    def start_position_timer():
        """Start a timer to periodically update position display"""
        nonlocal position_timer_running
        if position_timer_running:
            return  # Already running
        position_timer_running = True
        
        import time as time_module
        
        def timer_loop():
            while position_timer_running and edit_mode_active and state.is_playing:
                try:
                    update_position_display()
                except:
                    pass
                time_module.sleep(0.2)
        
        threading.Thread(target=timer_loop, daemon=True).start()
    
    def stop_position_timer():
        """Stop the position update timer"""
        nonlocal position_timer_running
        position_timer_running = False
    
    def play_edit_mode(e):
        if state.active_index < 0: return
        
        cand = state.candidates[state.active_index]
        v = get_video_control()
        if v:
            nonlocal current_playback_position, playback_start_time, playback_start_position
            # Use current position if within range, otherwise start from beginning
            if current_playback_position < cand.start or current_playback_position > cand.end:
                current_playback_position = cand.start
            
            # Record playback start time for position estimation
            import time as time_module
            playback_start_time = time_module.time()
            playback_start_position = current_playback_position
            
            # Seek to current position
            if hasattr(v, 'seek'): v.seek(int(current_playback_position * 1000))
            elif hasattr(v, 'jump_to'): v.jump_to(int(current_playback_position * 1000))
            
            # Play
            if hasattr(v, 'play'): v.play()
            state.is_playing = True
            
            # Start position tracking timer
            start_position_timer()
    
    def pause_edit_mode(e):
        """Pause video and stop position timer"""
        v = get_video_control()
        if v and hasattr(v, 'pause'):
            v.pause()
        state.is_playing = False
        stop_position_timer()

    # Range Editor Overlay
    range_overlay = ft.Container(
        content=ft.Column([
            # 1. Range Slider (Top) - For adjusting start/end
            ft.Container(
                content=range_slider,
                height=35,
                padding=ft.padding.only(left=10, right=10),
                alignment=ft.alignment.center
            ),
            
            # 2. Position Slider (Middle) - For playback position
            ft.Container(
                content=ft.Row([
                    position_label,
                    ft.Container(content=position_slider, expand=True),
                ], spacing=10),
                height=30,
                padding=ft.padding.only(left=10, right=10),
            ),
            
            # 3. Controls Area (Bottom)
            ft.Row([
                ft.Row([
                    ft.IconButton(ft.Icons.PLAY_ARROW, icon_color="white", icon_size=30, on_click=play_edit_mode, tooltip="現在位置から再生"),
                    ft.IconButton(ft.Icons.PAUSE, icon_color="white", icon_size=30, on_click=pause_edit_mode, tooltip="一時停止"),
                    ft.Icon(ft.Icons.VOLUME_UP, color="white", size=20),
                    ft.Slider(min=0, max=100, value=100, width=100, active_color="white", on_change=on_volume_change),
                ], spacing=5),
                
                range_label, 
                
                ft.Row([
                     ft.ElevatedButton("保存", icon=ft.Icons.CHECK, color="white", bgcolor="green", on_click=save_edit_mode),
                     ft.ElevatedButton("キャンセル", icon=ft.Icons.CLOSE, color="white", bgcolor="red", on_click=cancel_edit_mode),
                ], spacing=10)
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            
        ], spacing=5, alignment=ft.MainAxisAlignment.END), # Align content to bottom if needed, or just fill
        bgcolor="#CC000000",
        padding=10,
        border_radius=ft.border_radius.only(top_left=10, top_right=10),
        visible=False,
        bottom=0, left=0, right=0,
    )

    # Player Stack (Video + Overlay)
    player_stack = ft.Stack(expand=True)
    
    # Preview Container (Wraps Stack)
    player_container = ft.Container(
        content=player_stack,
        expand=True,
        bgcolor="black",
        border_radius=10,
        alignment=ft.alignment.center
    )
    
    # Placeholder black bg
    player_stack.controls.append(ft.Container(bgcolor="black", expand=True))
    player_stack.controls.append(range_overlay)
    
    candidates_list = ft.ListView(expand=True, spacing=2)
    
    export_btn = ft.ElevatedButton(
        "選択範囲をエクスポート (個別ファイル)", 
        icon=ft.Icons.SAVE, 
        disabled=True
    )

    # --- Event Handlers ---

    # --- Event Handlers ---

    def enter_edit_mode(idx):
        if idx < 0 or idx >= len(state.candidates): return
        
        # Backup
        cand = state.candidates[idx]
        state.active_index = idx
        editing_backup['start'] = cand.start
        editing_backup['end'] = cand.end
        
        # Calculate ROI (+/- 60s)
        roi_start = max(0.0, cand.start - 60.0)
        roi_end = min(state.duration, cand.end + 60.0)
        
        # Calculate ROI (+/- 60s)
        roi_start = max(0.0, cand.start - 60.0)
        roi_end = min(state.duration, cand.end + 60.0)
        
        # Create NEW RangeSlider to avoid state update issues
        global range_slider # Update global ref if needed
        range_slider = ft.RangeSlider(
            min=roi_start, 
            max=roi_end,
            start_value=cand.start,
            end_value=cand.end,
            label="{value}",
            on_change=on_range_change,
            on_change_end=on_range_change_end,
            overlay_color="transparent",
            active_color="orange",
            inactive_color="#4DFFFFFF",
            divisions=None # Continuous
        )

        # Replace in UI
        # range_overlay.content is Column
        # controls[0] is Container (Range Slider)
        # controls[1] is Container (Position Slider Row)
        range_overlay.content.controls[0].content = range_slider
        
        # Setup position slider to match range
        nonlocal current_playback_position, edit_mode_active
        current_playback_position = cand.start
        position_slider.min = cand.start
        position_slider.max = cand.end
        position_slider.value = cand.start
        position_label.value = format_time(cand.start)
        
        range_label.value = f"{format_time(cand.start)} - {format_time(cand.end)}"
        
        # Enable edit mode flag
        edit_mode_active = True
        
        # UI Transition
        right_pane.visible = False
        range_overlay.visible = True
        
        # Seek to start
        play_edit_mode(None)
        
        range_overlay.update()
        page.update()

    def exit_edit_mode(save: bool):
        nonlocal edit_mode_active
        edit_mode_active = False  # Disable edit mode
        
        if not save and state.active_index >= 0:
            # Restore
            cand = state.candidates[state.active_index]
            cand.start = editing_backup.get('start', cand.start)
            cand.end = editing_backup.get('end', cand.end)
            
        # Pause video when exiting edit mode
        v = get_video_control()
        if v and hasattr(v, 'pause'):
            v.pause()
        state.is_playing = False
            
        range_overlay.visible = False
        right_pane.visible = True
        
        # Re-render list to show updates (if saved)
        render_list_items()
        
        # Clear active selection visual?
        # Maybe keep it selected but stop editing mode.
        page.update()

    def set_active_candidate(idx):
        if idx < 0 or idx >= len(state.candidates): return
        state.active_index = idx
        # Just update internal state, do NOT open overlay automatically anymore
        pass


    # --- Event Handlers ---

    def on_export_click(e):
        if not state.video_path or not state.selected_indices:
            return
        
        save_file_picker.save_file(
            dialog_title="保存先 (この名前をベースに連番で保存されます)",
            file_name="highlight.mp4",
            allowed_extensions=["mp4"]
        )

    def on_save_result(e: ft.FilePickerResultEvent):
        if e.path:
            status_text.value = "エクスポート中 (個別ファイル)..."
            progress_bar.visible = True
            progress_bar.value = None
            export_btn.disabled = True
            page.update()
            threading.Thread(target=run_export, args=(e.path,)).start()

    save_file_picker.on_result = on_save_result
    export_btn.on_click = on_export_click

    def run_export(output_path):
        try:
            logging.info(f"Starting export to {output_path}")
            from logic.exporter import exporter, ExportSegment
            
            segments = []
            sorted_indices = sorted(list(state.selected_indices))
            for idx in sorted_indices:
                cand = state.candidates[idx]
                segments.append(ExportSegment(start=cand.start, end=cand.end))
            
            exporter.export_highlights(
                state.video_path, 
                output_path, 
                segments, 
                precise=True, 
                separate_files=True 
            )
            
            show_snack(f"保存完了: {output_path}周辺")
            reset_ui_state(exporting=False)
        except Exception as err:
            logging.error(f"Export error: {err}")
            show_snack(f"エラー: {err}", is_error=True)
            reset_ui_state(exporting=False)

    def reset_ui_state(exporting=False):
        status_text.value = "準備完了" if not exporting else "処理中..."
        progress_bar.visible = False
        progress_bar.value = 0
        progress_text.visible = False
        export_btn.disabled = len(state.selected_indices) == 0
        page.update()

    def show_snack(msg, is_error=False):
        page.snack_bar = ft.SnackBar(ft.Text(msg), bgcolor="red" if is_error else "green")
        page.snack_bar.open = True
        page.update()

    def on_check_change(e, idx):
        if e.control.value:
            state.selected_indices.add(idx)
            # If checking, maybe play it too? Optional.
        else:
            state.selected_indices.discard(idx)
        export_btn.disabled = len(state.selected_indices) == 0
        export_btn.update()
    
    def on_time_edit_str(e, idx, is_start):
        val_str = e.control.value
        try:
            parts = val_str.split(':')
            if len(parts) == 2:
                seconds = float(parts[0]) * 60 + float(parts[1])
            else:
                seconds = float(val_str)
            
            if idx < len(state.candidates):
                if is_start:
                    state.candidates[idx].start = seconds
                else:
                    state.candidates[idx].end = seconds
            
            if state.active_index == idx:
                 set_active_candidate(idx)
        except ValueError:
            pass

    def adjust_time(idx, is_start, delta):
        if idx < len(state.candidates):
            cand = state.candidates[idx]
            if is_start:
                cand.start = max(0, cand.start + delta)
            else:
                cand.end = max(cand.start, cand.end + delta)
            
            # Sync slider if this is active
            if state.active_index == idx:
                 set_active_candidate(idx)
            
            render_list_items()

    def play_candidate(idx):
        # Access the current Video control
        try:
            video_obj = player_stack.controls[0]
        except:
            return

        # Toggle Logic
        if state.active_index == idx and state.is_playing:
            # Already playing this candidate -> Pause
            if hasattr(video_obj, 'pause'):
                 video_obj.pause()
            state.is_playing = False
            render_list_items() # Update icon to Play
            return

        # Otherwise -> Play (Switch or Resume)
        state.active_index = idx
        
        cand = state.candidates[idx]
        
        # Seek
        if hasattr(video_obj, 'seek'):
            video_obj.seek(int(cand.start * 1000))
        elif hasattr(video_obj, 'jump_to'):
            video_obj.jump_to(int(cand.start * 1000))
        
        # Play
        if hasattr(video_obj, 'play'):
            video_obj.play()
            state.is_playing = True
        
        render_list_items() # Update icon to Pause
        
    import time as time_module
    
    def update_position_display():
        """Update position slider based on elapsed time since playback started"""
        nonlocal current_playback_position, playback_start_time, playback_start_position
        if not edit_mode_active or not state.is_playing:
            return
        
        if state.active_index < 0 or state.active_index >= len(state.candidates):
            return
        
        cand = state.candidates[state.active_index]
        
        # Calculate current position based on elapsed time
        elapsed = time_module.time() - playback_start_time
        pos_sec = playback_start_position + elapsed
        
        current_playback_position = pos_sec
        
        # Loop: if position exceeds end, seek back to start
        if pos_sec >= cand.end:
            v = get_video_control()
            if v:
                if hasattr(v, 'seek'): v.seek(int(cand.start * 1000))
                elif hasattr(v, 'jump_to'): v.jump_to(int(cand.start * 1000))
                current_playback_position = cand.start
                # Reset timing for loop
                playback_start_time = time_module.time()
                playback_start_position = cand.start
        
        # Update position slider (clamp to range)
        clamped_pos = max(cand.start, min(cand.end, current_playback_position))
        position_slider.value = clamped_pos
        position_label.value = format_time(clamped_pos)
        try:
            position_slider.update()
            position_label.update()
        except:
            pass

    def initialize_player(path):
        # Create NEW video control
        new_player = ft.Video(
            expand=True,
            fill_color="black",
            aspect_ratio=16/9,
            volume=100,
            autoplay=False, # Disable Auto play
            filter_quality=ft.FilterQuality.HIGH,
            muted=False,
            playlist_mode=True, 
            playlist=[ft.VideoMedia(path)]
        )
        
        # Stack: [Video, Overlay]
        player_stack.controls.clear()
        player_stack.controls.append(new_player)
        
        # Transparent Click Overlay for Play/Pause (Leaving bottom 100px for native controls)
        click_overlay = ft.Container(
            on_click=toggle_playback,
            bgcolor="transparent",
            # Absolute positioning to avoid layout shift
            left=0,
            right=0,
            top=0,
            bottom=100,
        )
        player_stack.controls.append(click_overlay)
        
        player_stack.controls.append(range_overlay)
        player_stack.update()
        state.is_playing = False # Sync with autoplay=False

    def play_clip_preview(idx):
        try:
            cand = state.candidates[idx]
            update_status("クリップ作成中... (数秒かかります)")
            
            from logic.exporter import exporter, ExportSegment
            import tempfile
            
            temp_dir = tempfile.gettempdir()
            # unique name to avoid conflict/caching
            preview_path = os.path.join(temp_dir, f"preview_{idx}_{int(cand.start)}_{int(cand.end)}.mp4")
            
            # Export single segment
            exporter.export_highlights(
                input_video=state.video_path,
                output_video=preview_path,
                segments=[ExportSegment(cand.start, cand.end)],
                precise=True,
                separate_files=False 
            )
            
            # Play it
            initialize_player(preview_path)
            update_status(f"クリップ再生中: {format_time(cand.start)} - {format_time(cand.end)}")
            
        except Exception as e:
            logging.error(f"Clip preview error: {e}")
            show_snack(f"クリップ再生エラー: {e}", is_error=True)
            update_status("エラー")

    def reset_time(idx):
        if idx < len(state.candidates):
            cand = state.candidates[idx]
            # Safety fallback: if original_start/end missing, use current values (no change)
            cand.start = getattr(cand, 'original_start', cand.start)
            cand.end = getattr(cand, 'original_end', cand.end)
            if state.active_index == idx:
                 set_active_candidate(idx)
            render_list_items()

    def toggle_sort(e):
        nonlocal sort_mode
        # Cycle: None -> desc -> asc -> None
        if sort_mode is None:
            sort_mode = 'desc'
        elif sort_mode == 'desc':
            sort_mode = 'asc'
        else:
            sort_mode = None
        render_list_items()

    def render_list_items():
        candidates_list.controls.clear()
        
        # Width definitions - Widened as requested
        w_check = 40
        w_play = 40 # New Play Button
        w_start_ctrl = 150 
        w_sep = 10
        w_end_ctrl = 150 
        w_time_block = w_start_ctrl + w_sep + w_end_ctrl
        w_reset = 50 
        w_time_block = w_start_ctrl + w_sep + w_end_ctrl 
        w_reset = 50 
        w_score = 120 # Widened for details
        w_edit = 50
        
        # Header Row
        header_row = ft.Container(
            content=ft.Row([
                ft.Text("選択", width=w_check, text_align=ft.TextAlign.CENTER, size=12, weight=ft.FontWeight.BOLD),
                ft.Text("再生", width=w_play, text_align=ft.TextAlign.CENTER, size=12, weight=ft.FontWeight.BOLD),
                ft.Text("範囲 (開始 ~ 終了)", width=w_time_block, text_align=ft.TextAlign.CENTER, size=12, weight=ft.FontWeight.BOLD),
                ft.Text("復元", width=w_reset, text_align=ft.TextAlign.CENTER, size=12, weight=ft.FontWeight.BOLD),
                ft.Container(
            content=ft.Row([
                ft.Text("スコア", size=12, weight=ft.FontWeight.BOLD),
                ft.IconButton(
                    ft.Icons.SWAP_VERT if sort_mode is None else (ft.Icons.ARROW_DOWNWARD if sort_mode == 'desc' else ft.Icons.ARROW_UPWARD),
                    icon_size=14,
                    width=20, height=20,
                    style=ft.ButtonStyle(padding=0),
                    on_click=toggle_sort,
                    tooltip="ソート切替"
                )
            ], spacing=2, alignment=ft.MainAxisAlignment.CENTER),
            width=w_score
        ),
                ft.Text("編集", width=w_edit, text_align=ft.TextAlign.CENTER, size=12, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
            bgcolor="#EEEEEE",
            padding=5,
            border_radius=5
        )
        candidates_list.controls.append(header_row)
        
        visible_indices = [i for i, c in enumerate(state.candidates) if c.score >= state.min_score]
        
        # Sort by score (if sort_mode is set)
        if sort_mode == 'desc':
            visible_indices.sort(key=lambda i: state.candidates[i].score, reverse=True)
        elif sort_mode == 'asc':
            visible_indices.sort(key=lambda i: state.candidates[i].score, reverse=False)
        
        results_header_text.value = f"検出結果 (表示: {len(visible_indices)} / 全体: {len(state.candidates)})"
        results_header_text.update()

        for i in visible_indices:
            cand = state.candidates[i]
            is_selected = i in state.selected_indices
            
            # Time Controls (Start)
            start_ctrl = ft.Row([
                ft.IconButton(ft.Icons.REMOVE, icon_size=16, width=30, height=30, style=ft.ButtonStyle(padding=0), on_click=lambda e, idx=i: adjust_time(idx, True, -1.0), tooltip="-1s"),
                ft.TextField(
                    value=format_time(cand.start),
                    width=70, text_size=13, content_padding=5, height=25,
                    on_change=lambda e, idx=i: on_time_edit_str(e, idx, True)
                ),
                ft.IconButton(ft.Icons.ADD, icon_size=16, width=30, height=30, style=ft.ButtonStyle(padding=0), on_click=lambda e, idx=i: adjust_time(idx, True, 1.0), tooltip="+1s"),
            ], spacing=1, width=w_start_ctrl, alignment=ft.MainAxisAlignment.CENTER)

            # Time Controls (End)
            end_ctrl = ft.Row([
                ft.IconButton(ft.Icons.REMOVE, icon_size=16, width=30, height=30, style=ft.ButtonStyle(padding=0), on_click=lambda e, idx=i: adjust_time(idx, False, -1.0), tooltip="-1s"),
                ft.TextField(
                    value=format_time(cand.end),
                    width=70, text_size=13, content_padding=5, height=25,
                    on_change=lambda e, idx=i: on_time_edit_str(e, idx, False)
                ),
                ft.IconButton(ft.Icons.ADD, icon_size=16, width=30, height=30, style=ft.ButtonStyle(padding=0), on_click=lambda e, idx=i: adjust_time(idx, False, 1.0), tooltip="+1s"),
            ], spacing=1, width=w_end_ctrl, alignment=ft.MainAxisAlignment.CENTER)

            # Score Details
            score_text = f"{cand.score:.2f}"
            detail_text = ""
            if hasattr(cand, 'details') and cand.details:
                top_class = cand.details.get("top_class", "")
                if top_class and top_class != "Unknown":
                    # Shorten some long names if needed
                    detail_text = top_class
            
            score_cell = ft.Column([
                ft.Text(score_text, size=13, weight=ft.FontWeight.BOLD),
                ft.Text(detail_text, size=10, color="grey", overflow=ft.TextOverflow.ELLIPSIS)
            ], spacing=0, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

            # Row Content
            row_content = ft.Container(
                content=ft.Row(
                    controls=[
                        # Checkbox
                        ft.Container(
                             content=ft.Checkbox(
                                value=is_selected,
                                on_change=lambda e, idx=i: on_check_change(e, idx)
                            ),
                            width=w_check, alignment=ft.alignment.center
                        ),
                        # Play Button
                        ft.Container(
                            content=ft.IconButton(
                                ft.Icons.PAUSE_CIRCLE_FILLED if (i == state.active_index and state.is_playing) else ft.Icons.PLAY_CIRCLE_FILL, 
                                icon_color="red" if (i == state.active_index and state.is_playing) else "green", 
                                icon_size=24,
                                width=30, height=30,
                                style=ft.ButtonStyle(padding=0),
                                on_click=lambda e, idx=i: play_candidate(idx),
                                tooltip="一時停止" if (i == state.active_index and state.is_playing) else "再生"
                            ),
                            width=w_play, alignment=ft.alignment.center
                        ),
                        # Time Block
                        ft.Container(
                            content=ft.Row([
                                start_ctrl,
                                ft.Text("~", width=w_sep, text_align=ft.TextAlign.CENTER),
                                end_ctrl
                            ], spacing=0, alignment=ft.MainAxisAlignment.CENTER),
                            width=w_time_block
                        ),
                        # Reset Button
                        ft.Container(
                            content=ft.IconButton(
                                ft.Icons.RESTORE, 
                                icon_size=18, 
                                tooltip="初期位置に戻す",
                                on_click=lambda e, idx=i: reset_time(idx)
                            ),
                            width=w_reset, alignment=ft.alignment.center
                        ),
                        # Score
                        ft.Container(
                            content=score_cell,
                            width=w_score, alignment=ft.alignment.center
                        ),
                        # Edit Button
                        ft.Container(
                            content=ft.IconButton(
                                ft.Icons.EDIT, 
                                icon_color="orange", 
                                icon_size=24, 
                                tooltip="詳細編集モード", 
                                on_click=lambda e, idx=i: enter_edit_mode(idx) # Click pencil -> Edit Mode
                            ),
                            width=w_edit, alignment=ft.alignment.center
                        )
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=0 
                ),
                padding=5,
                ink=True
            )
            
            item_container = ft.Container(
                content=row_content,
                bgcolor="#E6F4FF" if is_selected else "white", 
                border=ft.border.all(1, "blue" if is_selected else "grey300"), 
                border_radius=5,
                margin=ft.margin.only(bottom=2)
            )
            candidates_list.controls.append(item_container)
            
        page.update()

    def on_progress(p: float):
        progress_bar.value = p
        percentage = int(p * 100)
        progress_text.value = f"{percentage}%"
        progress_bar.update()
        progress_text.update()

    def run_analysis_thread(path):
        try:
            logging.info(f"Starting analysis for {path}")
            from logic.analyzer import analyzer
            
            wav_path = "temp_analysis.wav"
            analyzer.extract_audio(path, wav_path)
            
            update_status("AI解析中 (PANNs)...")
            
            candidates = analyzer.analyze_audio(
                wav_path, 
                padding=20.0, 
                progress_cb=on_progress,
                status_cb=update_status,
                batch_size=state.batch_size 
            )
            state.candidates = candidates
            # Save original times for reset fallback
            for c in state.candidates:
                c.original_start = c.start
                c.original_end = c.end
            state.selected_indices = set() # Default none
            
            cleanup_temp_file(wav_path)
            
            logging.info(f"Analysis complete: {len(candidates)} candidates")
            update_status(f"解析完了: {len(candidates)} 個の候補")
            finish_analysis(path)
            
        except Exception as e:
            logging.error(f"Analysis error: {e}", exc_info=True)
            show_snack(f"解析エラー: {str(e)}", is_error=True)
            update_status("エラーが発生しました")
            hide_progress()

    def update_status(msg):
        status_text.value = msg
        status_text.update()
    
    def finish_analysis(path):
        progress_bar.visible = False
        progress_text.visible = False
        
        # Init Player with Full Video Control BEFORE rendering list
        initialize_player(path)
        
        render_list_items()
        
        set_ui_locked(False) # Unlock
        page.update()
    
    def hide_progress():
        progress_bar.visible = False
        progress_text.visible = False
        page.update()

    def on_file_picked(e: ft.FilePickerResultEvent):
        if e.files:
            file_path = e.files[0].path
            if not file_path: return
                
            state.video_path = file_path
            
            # Get Duration
            try:
                # Lazy import to avoid startup hang
                try:
                    from moviepy import VideoFileClip
                except ImportError:
                    from moviepy.editor import VideoFileClip
                    
                clip = VideoFileClip(file_path)
                state.duration = clip.duration
                clip.close()
                logging.info(f"Video Duration: {state.duration}s")
            except Exception as e:
                logging.error(f"Failed to get duration: {e}")
                state.duration = 600.0 # Fallback
            
            status_text.value = f"準備中: {os.path.basename(file_path)} (長さ: {format_time(state.duration)})"
            progress_bar.visible = True
            progress_bar.value = 0
            progress_text.value = "0%"
            progress_text.visible = True
            
            state.candidates = []
            state.selected_indices = set()
            state.active_index = -1
            range_slider.disabled = True
            range_label.value = "未選択"
            range_overlay.visible = False
            range_overlay.update()
            
            render_list_items()
            page.update()
            
            set_ui_locked(True) # Lock UI during analysis
            threading.Thread(target=run_analysis_thread, args=(file_path,)).start()

    file_picker.on_result = on_file_picked

    # Layout Construction
    header = ft.Row([
        ft.Icon(ft.Icons.SMART_DISPLAY, size=30, color="purple"),
        ft.Text("クリップ抽出くん ベータ", size=24, weight=ft.FontWeight.BOLD)
    ])

    file_select_btn = ft.ElevatedButton(
        "動画ファイルを選択", 
        icon=ft.Icons.UPLOAD_FILE, 
        on_click=lambda _: file_picker.pick_files(allow_multiple=False, allowed_extensions=["mp4", "mov", "avi", "mkv"])
    )

    controls = ft.Row([
        file_select_btn,
        export_btn,
    ])
    
    def set_ui_locked(locked: bool):
        file_select_btn.disabled = locked
        score_slider.disabled = locked
        # candidates_list.disabled = locked # Flet ListView disabled might not block item clicks, but visual cue
        if locked:
            export_btn.disabled = True
            range_overlay.visible = False
            state.active_index = -1
        else:
            # Export btn state depends on selection
            export_btn.disabled = len(state.selected_indices) == 0
            
        page.update()
    
    slider_row = ft.Row([
        ft.Text("スコアフィルタ:", size=14, weight=ft.FontWeight.BOLD),
        ft.Container(score_slider, width=300),
        score_label
    ], alignment=ft.MainAxisAlignment.START)
    
    progress_row = ft.Row([progress_bar, progress_text], alignment=ft.MainAxisAlignment.START)
    
    # Left Preview
    left_pane = ft.Column([
        ft.Text("プレビュー", weight=ft.FontWeight.BOLD),
        player_container,
        ft.Divider(),
        # Range slider is now overlay, so removed container
    ], expand=True)
    
    # Right Table (Header + List)
    
    right_pane = ft.Column([
        results_header_text,
        candidates_list
    ], width=650)

    page.add(
        header,
        ft.Divider(),
        controls,
        slider_row,
        progress_row,
        status_text,
        ft.Divider(),
        ft.Row([left_pane, right_pane], expand=True)
    )
    
    if not check_ffmpeg():
        page.add(ft.Text("Error: FFmpeg not found!", color="red"))

if __name__ == "__main__":
    try:
        ft.app(target=main)
    except Exception as e:
        print(f"Error: {e}")
