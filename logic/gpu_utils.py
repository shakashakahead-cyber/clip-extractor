import subprocess
import re
import logging

def get_recommended_batch_size():
    """
    Detects GPU and recommends a batch size.
    Returns: int (batch_size)
    """
    try:
        # Use wmic to get Name and AdapterRAM
        # Note: AdapterRAM is often capped at 4GB (32-bit limit) on Windows WMI
        cmd = "wmic path win32_VideoController get Name, AdapterRAM"
        # Run with creationflags to avoid console window popping up if packaged
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore')
        
        gpus = []
        lines = result.strip().split('\n')
        
        logging.info(f"GPU Detection Result: {lines}")
        
        for line in lines[1:]:
            line = line.strip()
            if not line: continue
            
            # Heuristic parsing
            # Look for bytes number
            match = re.search(r'(\d{8,})', line)
            vram_mb = 0
            name = line
            
            if match:
                bytes_str = match.group(1)
                vram_bytes = int(bytes_str)
                vram_mb = vram_bytes / (1024**2)
                name = line.replace(bytes_str, "").strip()
            
            gpus.append((name.lower(), vram_mb))
            
        if not gpus:
            logging.warning("No GPU detected via WMI.")
            return 4 # Safe default

        # Logic
        # Prioritize dedicated GPUs
        max_score = 0
        best_batch = 4
        
        for name, vram in gpus:
            score = 0
            batch = 4
            
            # VRAM Base
            if vram > 10000: # Valid >10GB
                batch = 32
                score += 3
            elif vram > 6000: # Valid >6GB
                batch = 16
                score += 2
            elif vram >= 3500: # ~4GB (WMI Cap often lands here)
                batch = 16 
                score += 1
            
            # Name Heuristics (Override/Boost if WMI capped)
            if "nvidia" in name or "radeon" in name or "geforce" in name:
                score += 2
                if batch < 16: batch = 16 # At least 16 for dedicated
                
                # High end keywords?
                if "9070" in name or "4090" in name or "4080" in name or "7900" in name or "6900" in name:
                   if batch < 32: batch = 32
                elif "xt" in name: # Generic XT usually high end AMD
                   if batch < 32: batch = 32
                elif "ti" in name and batch == 32: # 3080 Ti etc
                   if batch < 32: batch = 32
            
            if "intel" in name or "uuhd" in name or "iris" in name:
                # Integrate, keep conservative unless VRAM proves otherwise
                score -= 1
                
            if score > max_score:
                max_score = score
                best_batch = batch
        
        logging.info(f"Auto-configured Batch Size: {best_batch}")
        return best_batch

    except Exception as e:
        logging.error(f"GPU Autodetect failed: {e}")
        return 4 # Safe fallback
