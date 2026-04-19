import os
from pathlib import Path

def check_dir(path):
    p = Path(path)
    if not p.exists():
        print(f"❌ {path} no existe")
        return
        
    files = list(p.glob("patch_*.pt"))
    print(f"📂 {path}: {len(files)} archivos")
    
    if not files:
        return
        
    indices = []
    for f in files:
        try:
            full_name = f.name
            idx_part = f.stem.split('_')[1]
            indices.append(int(idx_part))
        except:
            print(f"   ⚠️ Archivo raro: {f.name}")
            
    if indices:
        print(f"   Min ID: {min(indices)}")
        print(f"   Max ID: {max(indices)}")
        print(f"   Sample IDs: {indices[:5]} ... {indices[-5:]}")
        
    # Check specifically for the troublesome index
    problem_file = p / "patch_752223.pt"
    if problem_file.exists():
        print(f"   🚨 {problem_file.name} EXISTE!")
    else:
        print(f"   ✅ {problem_file.name} NO existe.")

print("--- Diagnóstico de Archivos ---")
check_dir("data/processed/patches_temporal_strict/train")
check_dir("data/processed/patches_temporal_strict/val")
