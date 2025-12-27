import os
import subprocess
from typing import List
from dataclasses import dataclass
from .utils import get_ffmpeg_path, cleanup_temp_file

@dataclass
class ExportSegment:
    start: float
    end: float

class Exporter:
    def __init__(self):
        pass

    def export_highlights(self, input_video: str, output_video: str, segments: List[ExportSegment], precise: bool = True, separate_files: bool = False):
        """
        Export selected segments to video file(s).
        separate_files: If True, exports inputs as separate files (base_001.mp4, etc).
        """
        if not segments:
            raise ValueError("No segments to export.")
            
        ffmpeg_path = get_ffmpeg_path()
        if not ffmpeg_path:
             raise FileNotFoundError("FFmpeg not found.")
             
        # Base folder and name
        output_dir = os.path.dirname(output_video)
        base_name = os.path.splitext(os.path.basename(output_video))[0]
        ext = os.path.splitext(output_video)[1]

        # 1. Create temporary segment files (or final files if separate)
        temp_files = []
        try:
            for i, seg in enumerate(segments):
                if separate_files:
                    # Final filename: base_001.mp4
                    out_file = os.path.join(output_dir, f"{base_name}_{i+1:03d}{ext}")
                else:
                    out_file = f"temp_seg_{i}.mp4"
                    temp_files.append(out_file)
                
                duration = seg.end - seg.start
                
                cmd = [ffmpeg_path, "-y"]
                
                if precise:
                    # Precise re-encoding
                    cmd.extend(["-ss", str(seg.start)])
                    cmd.extend(["-i", input_video])
                    cmd.extend(["-t", str(duration)])
                    cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"])
                    cmd.extend(["-c:a", "aac"])
                else:
                    # Stream copy
                    cmd.extend(["-ss", str(seg.start)])
                    cmd.extend(["-i", input_video])
                    cmd.extend(["-t", str(duration)])
                    cmd.extend(["-c", "copy"])
                
                cmd.append(out_file)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if not separate_files:
                # 2. Concatenate (only if not separate)
                list_file = "concat_list.txt"
                with open(list_file, "w", encoding="utf-8") as f:
                    for tf in temp_files:
                        f.write(f"file '{tf}'\n")
                
                concat_cmd = [
                    ffmpeg_path, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    output_video
                ]
                subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        finally:
            if not separate_files:
                if os.path.exists("concat_list.txt"):
                    os.remove("concat_list.txt")
                for tf in temp_files:
                    cleanup_temp_file(tf)

# Singleton instance
exporter = Exporter()
