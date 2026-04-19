import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, RandomCrop
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import time
import json
import numpy as np

# Add project root to path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports locales
from src.data.data_prop_improved import (
    SpreadDataset,
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    get_wind_indices,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from src.data.dataset_patch import PatchDataset

# ── Transforms on-the-fly (nivel módulo para que multiprocessing pueda picklarlo) ──
# Índices dinámicos de wind_u/wind_v en CHANNELS (single source of truth).
IDX_U, IDX_V = get_wind_indices(CHANNELS)
CROP  = 32  # Tamaño final del parche (32×32 px = 32×32 km)

class RandomAugment:
    """
    Augmentación aleatoria para train: flip/rot + CenterCrop(32).
    Corrige el vector de viento según la transformación geométrica.
    x: (T, C, H, W)  |  y: (1, H, W)
    """
    def __call__(self, x, y):
        import random
        op = random.randint(0, 7)
        if op == 0:   pass  # original
        elif op == 1: # rot90
            x = torch.rot90(x, 1, [2, 3]); y = torch.rot90(y, 1, [1, 2])
            x[:, IDX_U], x[:, IDX_V] = -x[:, IDX_V].clone(), x[:, IDX_U].clone()
        elif op == 2: # rot180
            x = torch.rot90(x, 2, [2, 3]); y = torch.rot90(y, 2, [1, 2])
            x[:, IDX_U] = -x[:, IDX_U]; x[:, IDX_V] = -x[:, IDX_V]
        elif op == 3: # rot270
            x = torch.rot90(x, 3, [2, 3]); y = torch.rot90(y, 3, [1, 2])
            x[:, IDX_U], x[:, IDX_V] = x[:, IDX_V].clone(), -x[:, IDX_U].clone()
        elif op == 4: # flipH
            x = torch.flip(x, [3]); y = torch.flip(y, [2])
            x[:, IDX_U] = -x[:, IDX_U]
        elif op == 5: # flipV
            x = torch.flip(x, [2]); y = torch.flip(y, [1])
            x[:, IDX_V] = -x[:, IDX_V]
        elif op == 6: # flipH + rot90
            x = torch.rot90(torch.flip(x, [3]), 1, [2, 3])
            y = torch.rot90(torch.flip(y, [2]), 1, [1, 2])
            x[:, IDX_U], x[:, IDX_V] = -x[:, IDX_V].clone(), -x[:, IDX_U].clone()
        elif op == 7: # flipV + rot90
            x = torch.rot90(torch.flip(x, [2]), 1, [2, 3])
            y = torch.rot90(torch.flip(y, [1]), 1, [1, 2])
            x[:, IDX_U], x[:, IDX_V] = x[:, IDX_V].clone(), x[:, IDX_U].clone()
        # Recorte central 32×32 (el fuego está centrado en el patch)
        h0 = (x.shape[-2] - CROP) // 2
        w0 = (x.shape[-1] - CROP) // 2
        x = x[..., h0:h0+CROP, w0:w0+CROP]
        y = y[..., h0:h0+CROP, w0:w0+CROP]
        return x, y

class CenterCrop32:
    """CenterCrop(32) para val/test — sin augmentación."""
    def __call__(self, x, y):
        h0 = (x.shape[-2] - CROP) // 2
        w0 = (x.shape[-1] - CROP) // 2
        return x[..., h0:h0+CROP, w0:w0+CROP], y[..., h0:h0+CROP, w0:w0+CROP]

class OffCenterCrop32:
    """
    Crop aleatorio en los BORDES del patch 224×224 (fuera del centro donde está el fuego).
    Genera negativos sintéticos sin necesidad de nuevos archivos en disco.
    Fuerza el crop a los cuadrantes exteriores (zona sin fuego).
    """
    def __call__(self, x, y):
        import random
        H, W = x.shape[-2], x.shape[-1]
        margin = CROP  # zona de exclusión alrededor del centro
        cx, cy = W // 2, H // 2
        # Elegir esquina aleatoria (top-left, top-right, bottom-left, bottom-right)
        corners = [
            (0, 0),
            (0, W - CROP),
            (H - CROP, 0),
            (H - CROP, W - CROP),
        ]
        h0, w0 = random.choice(corners)
        x = x[..., h0:h0+CROP, w0:w0+CROP]
        y = y[..., h0:h0+CROP, w0:w0+CROP]
        # Poner y a 0 (zona exterior = sin propagación garantizada)
        y = torch.zeros_like(y)
        return x, y

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true.flatten(), (y_pred > 0.5).flatten())

def calculate_f1(y_true, y_pred):
    return f1_score(y_true.flatten(), (y_pred > 0.5).flatten())

def calculate_iou(y_true, y_pred):
    # Intersection over Union for binary segmentation
    preds = (y_pred > 0.5).float()
    intersection = (preds * y_true).sum()
    union = preds.sum() + y_true.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
import src.config as config  # Importar config centralizada

# Configuración básica
def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar Modelo de Propagación de Fuego")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split_strategy", type=str, default="temporal", choices=["temporal", "spatial"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0, help="Número de workers para DataLoader (0 para evitar hangs en macOS)")
    parser.add_argument("--crop_size", type=int, default=224, help="Tamaño del recorte centrado en el fuego")
    return parser.parse_args()



# --- NORMALIZACIÓN (stats centralizadas en data/processed/spread_stats.json) ---
STATS = load_default_stats()

def normalize_batch(x):
    """Normaliza batch (B, T, C, H, W) usando stats centralizadas."""
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    progress_bar = tqdm(loader, desc="Training")
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        
        # ⚡ NORMALIZACIÓN
        x = normalize_batch(x)
        
        optimizer.zero_grad()
        outputs = model(x)
        
        # RobustFireSpreadModel devuelve un dict
        if isinstance(outputs, dict):
            pred = outputs['spread_probability']
        else:
            pred = outputs
        
        # Output shape might be (B, 1, H, W) or (B, H, W)
        # Ensure compatibility
        if pred.shape != y.shape:
            pred = pred.view_as(y)
            
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Metrics
        with torch.no_grad():
            iou = calculate_iou(y, pred)
            total_iou += iou
            pmean = pred.mean().item()
            pmax = pred.max().item()
            tsum = y.sum().item()
            
        progress_bar.set_postfix(
            loss=f"{loss.item():.3f}", 
            iou=f"{iou:.3f}", 
            pmax=f"{pmax:.3f}",
            tsum=f"{tsum:.0f}"
        )
        
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_iou

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = normalize_batch(x)
            outputs = model(x)
            
            if isinstance(outputs, dict):
                pred = outputs['spread_probability']
            else:
                pred = outputs
            
            if pred.shape != y.shape:
                pred = pred.view_as(y)
                
            loss = criterion(pred, y)
            total_loss += loss.item()
            total_iou += calculate_iou(y, pred)
            
    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / len(loader)
    return avg_loss, avg_iou


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"🚀 Iniciando entrenamiento en {device}")
    
    # 1. Cargar Datacube (YA NO ES NECESARIO con parches)
    # import xarray as xr
    # datacube = xr.open_dataset(config.DATACUBE_PATH) 
    
    # 3. Datasets & Loaders
    from src.data.dataset_patch import PatchDataset

    print("📦 Cargando Datasets desde parches pre-generados (.pt)...")

    # Auto-detectar si existe el nuevo dataset spread_32_all (pos+neg, ya en 32×32)
    from pathlib import Path as _Path
    from torch.utils.data import WeightedRandomSampler
    base_all  = "data/processed/patches/spread_32_all"
    base_orig = "data/processed/patches/spread_224_fires_only"

    if _Path(f"{base_all}/train").exists() and len(list(_Path(f"{base_all}/train").glob("*.pt"))) > 10:
        # ─— Nuevo dataset: 32×32, pos+neg, sin cropping adicional ──
        print("📂 Usando spread_32_all (pos+neg, 32×32 pre-guardado)")
        train_dir = f"{base_all}/train"
        val_dir   = f"{base_all}/val"

        # Augmentación manual (los patches ya son 32×32, rotar/flip sin crop adicional)
        class AugmentOnly:
            """Como RandomAugment pero sin CenterCrop (patches ya son 32×32)."""
            def __call__(self, x, y):
                import random
                op = random.randint(0, 7)
                if op == 0: pass
                elif op == 1:
                    x = torch.rot90(x,1,[2,3]); y = torch.rot90(y,1,[1,2])
                    x[:,IDX_U], x[:,IDX_V] = -x[:,IDX_V].clone(), x[:,IDX_U].clone()
                elif op == 2:
                    x = torch.rot90(x,2,[2,3]); y = torch.rot90(y,2,[1,2])
                    x[:,IDX_U] = -x[:,IDX_U]; x[:,IDX_V] = -x[:,IDX_V]
                elif op == 3:
                    x = torch.rot90(x,3,[2,3]); y = torch.rot90(y,3,[1,2])
                    x[:,IDX_U], x[:,IDX_V] = x[:,IDX_V].clone(), -x[:,IDX_U].clone()
                elif op == 4:
                    x = torch.flip(x,[3]); y = torch.flip(y,[2])
                    x[:,IDX_U] = -x[:,IDX_U]
                elif op == 5:
                    x = torch.flip(x,[2]); y = torch.flip(y,[1])
                    x[:,IDX_V] = -x[:,IDX_V]
                elif op == 6:
                    x = torch.rot90(torch.flip(x,[3]),1,[2,3])
                    y = torch.rot90(torch.flip(y,[2]),1,[1,2])
                    x[:,IDX_U], x[:,IDX_V] = -x[:,IDX_V].clone(), -x[:,IDX_U].clone()
                elif op == 7:
                    x = torch.rot90(torch.flip(x,[2]),1,[2,3])
                    y = torch.rot90(torch.flip(y,[1]),1,[1,2])
                    x[:,IDX_U], x[:,IDX_V] = x[:,IDX_V].clone(), x[:,IDX_U].clone()
                return x, y

        train_ds = PatchDataset(train_dir, transform=AugmentOnly())
        val_ds   = PatchDataset(val_dir,   transform=None)

        # WeightedRandomSampler usando el flag is_positive guardado en cada .pt
        weights = []
        for fp in sorted(_Path(train_dir).glob("sample_*_orig.pt")):
            d = torch.load(fp, weights_only=False)
            weights.append(7.0 if d.get('is_positive', True) else 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers)
        n_pos = sum(1 for w in weights if w == 7.0)
        n_neg = len(weights) - n_pos
        print(f"🟥 WeightedSampler: {n_pos} pos (w=7) + {n_neg} neg (w=1) en {train_dir}")

    else:
        # ─— Dataset original: solo positivos 224×224 → CenterCrop 32×32 on-the-fly ──
        print("📂 Usando spread_224_fires_only (solo positivos, crop on-the-fly)")
        train_dir = f"{base_orig}/train"
        val_dir   = f"{base_orig}/val"
        train_pos_ds = PatchDataset(train_dir, transform=RandomAugment())
        val_ds       = PatchDataset(val_dir,   transform=CenterCrop32())
        train_ds     = train_pos_ds
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        print(f"✅ Train: {len(train_ds)} muestras | Crop on-the-fly 224→{CROP}px")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    
    # 4. Modelo
    print("🧠 Inicializando Modelo RobustFireSpreadModel...")
    # Input channels: Features + 1 (Fire Mask)
    in_channels = len(CHANNELS) + 1 
    
    # AJUSTE CRÍTICO PARA 16x16:
    # Usamos solo 2 niveles de profundidad [64, 128]
    # Nivel 0: 16x16
    # Nivel 1: 8x8 (Bottleneck)
    # Si usáramos 3 niveles, bajaría a 4x4 (muy poco detalle espacial sin skip connections)
    model = RobustFireSpreadModel(input_channels=in_channels, hidden_dims=[64, 128]).to(device)
    
    # 5. Optimizador
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # --- LOSS FUNCTION (Focal + Dice) ---
    # BCE ponderada puede hacer que el modelo converja a una constante baja (ej. 0.17).
    # Focal Loss fuerza al modelo a enfocarse en los ejemplos "difíciles" (los píxeles de fuego minoritarios).
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.9, gamma=2.0):
            super().__init__()
            self.alpha = alpha     # Peso para la clase positiva (fuego)
            self.gamma = gamma     # Factor de focalización

        def forward(self, inputs, targets):
            inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
            # p_t: Probabilidad asignada a la clase correcta
            p_t = inputs * targets + (1 - inputs) * (1 - targets)
            # alpha_t: Factor de peso
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            
            focal_loss = -alpha_t * (1 - p_t)**self.gamma * torch.log(p_t)
            return focal_loss.mean()
            
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1):
            super(DiceLoss, self).__init__()
            self.smooth = smooth

        def forward(self, inputs, targets):
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            intersection = (inputs * targets).sum()                            
            dice = 1 - (2.*intersection + self.smooth)/(inputs.sum() + targets.sum() + self.smooth)  
            return dice
    
    class FocalDiceLoss(nn.Module):
        def __init__(self, alpha=0.9, gamma=2.0):
            super().__init__()
            self.focal = FocalLoss(alpha, gamma)
            self.dice = DiceLoss()

        def forward(self, inputs, targets):
            return self.focal(inputs, targets) + self.dice(inputs, targets)

    criterion = FocalDiceLoss(alpha=0.9, gamma=2.0)
    print(f"⚖️ Usando Focal Loss (alpha=0.9, gamma=2.0) + Dice Loss")
    print(f"   -> Mejor balance precision/recall en test.")


    
    # 6. Loop
    best_val_iou = 0.0

    for epoch in range(args.epochs):
        print(f"\n🏷️  Epoch {epoch+1}/{args.epochs}")

        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        print(f"   Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"   Val Loss:   {val_loss:.4f} | IoU: {val_iou:.4f}")


        
        # Checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "models/best_spread_model.pth")
            print("   💾 Modelo guardado (Mejor IoU)")
            
    print("✨ Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
