import shutil
import subprocess
import os
import sys

APP_NAME = "VideoHighlightExtractor"

def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.getcwd() # Or os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_user_data_path(filename: str = "") -> str:
    """
    Get path to user data directory (AppData/Roaming/APP_NAME).
    Create directory if it doesn't exist.
    """
    app_data = os.getenv('APPDATA')
    if not app_data:
        app_data = os.path.expanduser("~") # Fallback to user home
        
    data_dir = os.path.join(app_data, APP_NAME)
    
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
        except OSError:
            pass # Should log this but simple pass for now
            
    if filename:
        return os.path.join(data_dir, filename)
    return data_dir


def check_ffmpeg():
    """Check if ffmpeg is available in the system path."""
    return shutil.which("ffmpeg") is not None

def get_ffmpeg_path():
    """Return the path to the ffmpeg executable. Prioritize bundled."""
    # Check bundled first
    bundled_path = get_resource_path("ffmpeg.exe")
    if os.path.exists(bundled_path):
        return bundled_path
        
    return shutil.which("ffmpeg")

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h >= 1:
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return f"{int(m):02d}:{int(s):02d}"

def cleanup_temp_file(filepath: str):
    """Safely remove a temporary file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up {filepath}: {e}")
