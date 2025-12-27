import PyInstaller.__main__
import os
import shutil

APP_NAME = "ClipExtractorBeta"
ENTRY_POINT = "main.py"

def build():
    print("Building EXE...")
    
    # 1. clean dist/build
    if os.path.exists("dist"): shutil.rmtree("dist")
    if os.path.exists("build"): shutil.rmtree("build")

    # 2. Define data to include
    # Format: "source;dest" for Windows
    datas = [
        ("panns_classes.txt", "."),
        ("*.onnx", "."),
        ("*.onnx.data", "."),
        # ("ffmpeg.exe", "."), # Removed to avoid GPL distribution issues. User provides ffmpeg.
    ]
    
    add_data_args = []
    for src, dst in datas:
        add_data_args.append(f"--add-data={src};{dst}")

    # 3. Run PyInstaller
    args = [
        ENTRY_POINT,
        f"--name={APP_NAME}",
        "--clean",
        "--noconfirm",
        # "--windowed", # Hides console. USEFUL FOR RELEASE. 
        # For Beta/Debug, maybe keep console? User didn't specify, but "public" usually means --windowed.
        # However, Flet apps sometimes need console if not packaged correctly, but standard is windowed.
        # I'll enable windowed but users can check log file if I added file logging.
        "--windowed", 
        "--onedir", # Folder mode - faster startup than --onefile
        # Exclude heavy unused libs if possible (optional)
    ] + add_data_args

    print(f"Running PyInstaller with args: {args}")
    PyInstaller.__main__.run(args)
    
    print("Build Complete. Check 'dist' folder.")

if __name__ == "__main__":
    build()
