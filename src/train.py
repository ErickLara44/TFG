import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import config
from src.data.data_ignition_improved import PrecomputedIgnitionDataset, DEFAULT_FEATURE_VARS
from src.models.ignition import RobustFireIgnitionModel

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        if torch.isnan(x).any():
            print("❌ Error: NaN detectado en input X")
            continue
        if torch.isnan(y).any():
            print("❌ Error: NaN detectado en target Y")
            continue
        
        optimizer.zero_grad()
        
        # Forward
        # Model returns: ignition_prob, confidence, risk_dist
        # We only care about ignition_prob for basic training for now
        outputs = model(x)
        pred_ignition = outputs['ignition']
        
        # Loss
        loss = criterion(pred_ignition, y.squeeze())
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            pred_ignition = outputs['ignition']
            
            loss = criterion(pred_ignition, y.squeeze())
            total_loss += loss.item()
            
            # Accuracy
            predicted = (torch.sigmoid(pred_ignition) > 0.5).float()
            correct += (predicted == y.squeeze()).sum().item()
            total += y.size(0)
            
    return total_loss / len(loader), correct / total

def main():
    print("🚀 Iniciando configuración de entrenamiento...")
    
    # Config
    DEVICE = torch.device(config.TRAINING_CONFIG['device'] if torch.cuda.is_available() or torch.backends.mps.is_available() else 'cpu')
    print(f"   Device: {DEVICE}")
    
    # Data Paths
    PATCHES_DIR = config.DATA_DIR / "processed" / "patches"
    if not PATCHES_DIR.exists():
        print(f"❌ Error: No se encontraron parches en {PATCHES_DIR}")
        print("   Ejecuta scripts/precompute_data.py primero.")
        sys.exit(1)
        
    # Datasets
    # Load Stats
    STATS_PATH = config.DATA_DIR / 'processed' / 'iberfire_normalization_stats.pkl'
    stats = None
    if STATS_PATH.exists():
        import pickle
        print(f"📊 Cargando estadísticas de normalización desde {STATS_PATH}")
        with open(STATS_PATH, 'rb') as f:
            stats = pickle.load(f)
    else:
        print("⚠️ Advertencia: No se encontraron estadísticas de normalización. El entrenamiento podría ser inestable.")

    # Nota: Necesitamos saber cuántos archivos hay para crear los índices
    # Esto asume que los archivos son patch_0.pt, patch_1.pt, ...
    # Una mejor implementación leería los archivos disponibles
    
    print("📂 Cargando datasets pre-computados...")
    datasets = {}
    for split in ['train', 'val']:
        split_dir = PATCHES_DIR / split
        n_files = len(list(split_dir.glob("patch_*.pt")))
        if n_files == 0:
            print(f"⚠️ No hay datos para {split}")
            continue
            
        datasets[split] = PrecomputedIgnitionDataset(
            patches_dir=split_dir,
            indices=list(range(n_files)),
            mode="convlstm",
            stats=stats
        )
        print(f"   {split}: {len(datasets[split])} muestras")
        
    if 'train' not in datasets:
        print("❌ No hay dataset de entrenamiento!")
        sys.exit(1)

    # Dataloaders
    train_loader = DataLoader(datasets['train'], batch_size=config.TRAINING_CONFIG['batch_size'], shuffle=True)
    
    val_loader = None
    if 'val' in datasets:
        val_loader = DataLoader(datasets['val'], batch_size=config.TRAINING_CONFIG['batch_size'], shuffle=False)
    else:
        print("⚠️ Advertencia: No hay dataset de validación. Se saltará la validación.")
    
    # Model
    print("🧠 Inicializando modelo...")
    model = RobustFireIgnitionModel(
        num_input_channels=len(DEFAULT_FEATURE_VARS),
        temporal_context=config.MODEL_CONFIG['temporal_context'],
        hidden_dims=[64, 128] # Default or from config
    ).to(DEVICE)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    # Loop
    epochs = config.TRAINING_CONFIG['epochs']
    best_val_loss = float('inf')
    
    print(f"\n🔥 Comenzando entrenamiento por {epochs} épocas...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"   Train Loss: {train_loss:.4f}")
        
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
            # Save best (only if validating)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = config.MODELS_DIR / "best_ignition_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"   💾 Modelo guardado en {save_path}")
        else:
            # Save every epoch if no validation
            save_path = config.MODELS_DIR / "last_ignition_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Modelo guardado en {save_path}")

if __name__ == "__main__":
    main()
