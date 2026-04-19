
import sys
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ignition import RobustFireIgnitionModel
from src.data.data_ignition_improved import PrecomputedIgnitionDataset

def main():
    print("🚀 Iniciando Debug Script...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_dir = "data/processed/patches_temporal_strict_FULL/train"
    train_dir = Path(data_dir)
    train_files = sorted(list(train_dir.glob("patch_*.pt")))[:100] # Solo 100 archivos
    
    print(f"📂 Archivos encontrados: {len(train_files)}")
    
    # Indices
    indices = [int(f.stem.split('_')[1]) for f in train_files]
    
    # Dataset
    print("📦 Creando Dataset...")
    ds = PrecomputedIgnitionDataset(train_dir, indices=indices, stats=None)
    
    # Loader
    print("🚛 Creando DataLoader (num_workers=0)...")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Iterar
    print("🔄 Iterando DataLoader...")
    start_time = time.time()
    for i, (x, y) in enumerate(loader):
        print(f"   Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
        x = x.to(device)
        y = y.to(device)
        print(f"   Batch {i} movido a device")
        
        if i >= 2: break
        
    print("🏗️ Inicializando Modelo, Optimizador y Loss...")
    T, C, H, W = 3, 18, 64, 64 # Harcoded
    model = RobustFireIgnitionModel(C, T, hidden_dims=[64, 64], dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    
    print("🔄 Iterando DataLoader con BACKWARD pass...")
    start_time = time.time()
    
    for i, (x, y) in enumerate(loader):
        print(f"   Batch {i} Start")
        x = x.to(device)
        y = y.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(x)
        logits = outputs['ignition']
        
        # Loss
        loss = criterion(logits, y)
        print(f"   Batch {i} Loss: {loss.item()}")
        
        # Backward
        loss.backward()
        print(f"   Batch {i} Backward OK")
        
        # Step
        optimizer.step()
        print(f"   Batch {i} Optimizer OK")
        
        if i >= 2: break
        
    print(f"✅ Full Training Step OK ({time.time() - start_time:.2f}s)")
    print("🎉 Debug completado.")

if __name__ == "__main__":
    main()
