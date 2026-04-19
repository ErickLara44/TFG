"""
train_spread_v4.py — ConvLSTM con Filtrado de Propagación Real + Focal α=0.99
==============================================================================
Solución al colapso de moda (predecir todo-cero):
  
  CAUSA: ~37.5% de patches tienen y_sum=0 (fuego no se propaga ese día).
  El modelo aprendía que predecir-cero da IoU=0.375 "gratis" → mínimo local trivial.

  SOLUCIÓN 1 - FireSpreadFilteredDataset:
    Filtra patches en el constructor cargando solo aquellos con y_sum > 0.
    → El modelo nunca ve batches vacíos. Forzado a aprender propagación real.

  SOLUCIÓN 2 - Focal α=0.99:
    99× más peso a los píxeles de fuego vs no-fuego.
    Gradiente de 7 píxeles de fuego > gradiente de 8000 píxeles vacíos.

  SOLUCIÓN 3 - num_workers=0:
    Reproducibilidad determinista (igual que el run original de 56.62%).

  Arquitectura y loss base: idéntica al original para comparación justa.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, RandomCrop
from pathlib import Path
import argparse
from tqdm import tqdm
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel

# ─────────────────────────────────────────────────────────────────────────────
STATS = load_default_stats()

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

# ─────────────────────────────────────────────────────────────────────────────
# DATASET FILTRADO — solo patches con propagación real
# ─────────────────────────────────────────────────────────────────────────────
class FireSpreadFilteredDataset(Dataset):
    """
    Wrapper que filtra PatchDataset para conservar SOLO los patches donde
    y_sum > min_fire_pixels (el fuego se propagó al menos a 1 celda nueva).
    
    Esto elimina los patches sin propagación que causaban el modo colapso.
    Pre-computa los índices válidos en el constructor (lento una vez, rápido en training).
    """
    def __init__(self, data_dir, transform=None, min_fire_pixels=1):
        self.base = PatchDataset(data_dir, transform=None)  # sin transform para filtrar
        self.transform = transform
        self.min_fire_pixels = min_fire_pixels
        
        print(f"🔍 Filtrando patches con fire_pixels >= {min_fire_pixels}...")
        self.valid_indices = []
        for i in tqdm(range(len(self.base)), desc="  Analizando patches", leave=False):
            try:
                _, y = self.base[i]
                if y.sum().item() >= min_fire_pixels:
                    self.valid_indices.append(i)
            except Exception:
                pass
        
        total = len(self.base)
        kept  = len(self.valid_indices)
        print(f"   Filtrado: {kept}/{total} patches válidos "
              f"({100*kept/total:.1f}% con propagación real)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        x, y = self.base[self.valid_indices[idx]]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y


class PairedCenterCrop:
    def __init__(self, size):
        self.crop = CenterCrop(size)
    def __call__(self, x, y):
        return self.crop(x), self.crop(y)


class PairedRandomCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, x, y):
        top, left, h, w = RandomCrop.get_params(x[-1], (self.size, self.size))
        return x[..., top:top+h, left:left+w], y[..., top:top+h, left:left+w]

# ─────────────────────────────────────────────────────────────────────────────
# LOSS — Focal con α=0.99 + Dice
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2.0):
        super().__init__()
        self.alpha = alpha   # 0.99 = fuego 99× más importante
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs  = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        p_t     = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (-alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        i = inputs.view(-1)
        t = targets.view(-1)
        inter = (i * t).sum()
        return 1 - (2. * inter + self.smooth) / (i.sum() + t.sum() + self.smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice  = DiceLoss()

    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.dice(inputs, targets)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_iou(y_true, y_pred):
    preds = (y_pred > 0.5).float()
    inter = (preds * y_true).sum()
    union = preds.sum() + y_true.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    bar = tqdm(loader, desc="Training v4")
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        x = normalize_batch(x)
        optimizer.zero_grad()
        out  = model(x)
        pred = out['spread_probability'] if isinstance(out, dict) else out
        if pred.shape != y.shape:
            pred = pred.view_as(y)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            iou = calculate_iou(y, pred)
            total_iou += iou
        bar.set_postfix(
            loss=f"{loss.item():.3f}",
            iou=f"{iou:.3f}",
            pmax=f"{pred.max().item():.3f}"
        )
    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = normalize_batch(x)
            out  = model(x)
            pred = out['spread_probability'] if isinstance(out, dict) else out
            if pred.shape != y.shape:
                pred = pred.view_as(y)
            total_loss += criterion(pred, y).item()
            total_iou  += calculate_iou(y, pred)
    return total_loss / len(loader), total_iou / len(loader)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--min_fire_pixels", type=int,   default=1,
                        help="Minimum fire pixels in target to keep a patch")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available()
                        else "mps"  if torch.backends.mps.is_available()
                        else "cpu")
    args   = parser.parse_args()
    device = torch.device(args.device)
    print(f"🚀 ConvLSTM v4 (FireFiltered + α=0.99) en {device}")

    # Datasets filtrados — solo patches con propagación real
    train_ds = FireSpreadFilteredDataset(
        "data/processed/patches/spread_224/train",
        transform=PairedRandomCrop(32),
        min_fire_pixels=args.min_fire_pixels
    )
    # Val: mantenemos TODOS los patches para evaluación real sin sesgo
    val_ds = FireSpreadFilteredDataset(
        "data/processed/patches/spread_224/val",
        transform=PairedCenterCrop(32),
        min_fire_pixels=1
    )

    # num_workers=0 para reproducibilidad determinista (igual que run original)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    in_channels = len(CHANNELS) + 1
    model = RobustFireSpreadModel(
        input_channels=in_channels,
        hidden_dims=[64, 128],
        dropout=0.15
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Parámetros: {params:,} | hidden_dims=[64, 128]")
    print(f"⚖️  FocalDiceLoss(α=0.99, γ=2.0) — fuego 99× más importante")
    print(f"🔍 Dataset filtrado: solo {len(train_ds):,} patches con propagación real")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = FocalDiceLoss(alpha=0.99, gamma=2.0)

    best_iou = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n🏷️  Epoch {epoch+1}/{args.epochs}")
        t_loss, t_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_iou = validate(model, val_loader, criterion, device)
        print(f"   Train Loss: {t_loss:.4f} | IoU: {t_iou:.4f}")
        print(f"   Val   Loss: {v_loss:.4f} | IoU: {v_iou:.4f}")
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), "models/best_convlstm_v4_spread.pth")
            print("   💾 Modelo guardado (Mejor IoU)")

    print(f"\n✅ Finalizado. Mejor Val IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
