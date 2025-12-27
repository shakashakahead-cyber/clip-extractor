import os
import urllib.request
from pathlib import Path
import torch
import numpy as np
import panns_inference
from panns_inference import AudioTagging
import onnx
import onnxruntime

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {target_path}...")
        urllib.request.urlretrieve(url, target_path)
        print("Done.")

def export_onnx_model(batch_size: int = 32):
    print("Setting up PANNs environment...")
    
    # PANNs default data directory
    home_dir = str(Path.home())
    panns_input_dir = os.path.join(home_dir, 'panns_data')
    if not os.path.exists(panns_input_dir):
        os.makedirs(panns_input_dir)
    
    # 1. Download Class Labels
    csv_url = 'https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv'
    csv_path = os.path.join(panns_input_dir, 'class_labels_indices.csv')
    download_file(csv_url, csv_path)
    
    # 2. Download Model Checkpoint (Standard 32k model)
    # Cnn14_mAP=0.431.pth
    model_url = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
    model_path = os.path.join(panns_input_dir, 'Cnn14_mAP=0.431.pth')
    download_file(model_url, model_path)
    
    print("Initializing PANNs AudioTagging model...")
    device = "cpu"
    
    # Patch os.system to avoid wget error from panns_inference
    original_system = os.system
    os.system = lambda x: 0 
    
    try:
        # Patch torch.load to handle weights_only=False (needed for PANNs checkpoint)
        original_load = torch.load
        
        def safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                 kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
            
        torch.load = safe_load
        
        print(f"Loading checkpoint: {model_path}")
        at = AudioTagging(checkpoint_path=model_path, device=device)
    finally:
        # Restore
        torch.load = original_load
        os.system = original_system

    # Define dummy input with FIXED batch size
    # This ensures the graph is fully optimized for this specific size (DirectML Stability)
    dummy_input = torch.randn(batch_size, 32000 * 10) 
    
    # Disable SpecAugmentor for export (causes dynamic shape errors and not needed for inference)
    import torch.nn as nn
    if hasattr(at.model, 'spec_augmenter'):
        print("Disabling SpecAugmentor for export...")
        at.model.spec_augmenter = nn.Identity()
    
    at.model.eval()

    # Save class labels
    # Read from downloaded CSV
    import csv
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        # format: index, mid, display_name
        labels = [row[2] for row in reader]
        
    with open("panns_classes.txt", "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")
    print("Saved panns_classes.txt")

    print(f"Exporting model to ONNX (Batch Size: {batch_size})...")
    
    output_path = f"Cnn14_batch{batch_size}.onnx" # Output name with batch size
    
    # Define Wrapper to ensure clean single output
    class PANNsWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # PANNs Cnn14 returns a dict: {'clipwise_output': ..., 'framewise_output': ...}
            # We only need clipwise_output for this app
            out = self.model(x)
            if isinstance(out, dict):
                return out['clipwise_output']
            return out

    wrapped_model = PANNsWrapper(at.model)
    wrapped_model.eval()
    
    # Export with FIXED shapes (No dynamic axes for batch) to prevent DirectML errors
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # Removed dynamic_axes to force fixed batch size compilation
    )
    print(f"ONNX model exported to {output_path}")
    
    # Verify
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")
    
    return output_path

if __name__ == "__main__":
    try:
        # Default run
        export_onnx_model(32)
    except Exception as e:
        print(f"Error: {e}")
