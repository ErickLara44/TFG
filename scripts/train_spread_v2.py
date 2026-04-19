"""
train_spread_v2.py — ConvLSTM con Data Augmentation Mejorada
=============================================================
Mejoras sobre train_spread.py (original):
  - Añadidas augmentaciones espaciales sincronizadas x/y:
      * Random Horizontal Flip (p=0.5)
      * Random Vertical Flip (p=0.5)
      * Random 90° Rotation (0, 90, 180, 270)

  Todo lo demás es idéntico al original:
  - Arquitectura: hidden_dims=[64, 128] (sin cambios)
  - Pérdida: FocalDiceLoss(alpha=0.9, gamma=2.0) (sin cambios)
  - Optimizer: AdamW + CosineAnnealingLR
  - Sin Physics Loss, sin Channel Dropout
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    get_wind_indices,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel

# ─────────────────────────────────────────────────────────────────────────────
# STATS centralizados (data/processed/spread_stats.json)
STATS = load_default_stats()

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

# ─────────────────────────────────────────────────────────────────────────────
# LOSS (idéntica al original)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (-alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice  = DiceLoss()

    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.dice(inputs, targets)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTED TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
# Channel indices in the (T, C, H, W) tensor — dinámicos desde CHANNELS
_WIND_U, _WIND_V = get_wind_indices(CHANNELS)

def _rotate_wind(x, k):
    """
    Rotate the wind vector (wind_u, wind_v) by k*90° CCW to match
    the spatial rotation applied by torch.rot90(x, k, [-2,-1]).
    Rotation matrix for k*90° CCW:
      k=1: (u, v) → (-v,  u)
      k=2: (u, v) → (-u, -v)
      k=3: (u, v) → ( v, -u)
    """
    x = x.clone()
    u = x[:, _WIND_U].clone()
    v = x[:, _WIND_V].clone()
    if k == 1:
        x[:, _WIND_U] = -v
        x[:, _WIND_V] =  u
    elif k == 2:
        x[:, _WIND_U] = -u
        x[:, _WIND_V] = -v
    elif k == 3:
        x[:, _WIND_U] =  v
        x[:, _WIND_V] = -u
    return x


class PairedRandomCropAug:
    """
    RandomCrop + Flips + Rotations 90°, sincronizados entre x e y.
    Los canales vectoriales wind_u / wind_v se transforman físicamente
    para mantener coherencia direccional tras cada operación espacial.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, x, y):
        from torchvision.transforms import RandomCrop
        top, left, h, w = RandomCrop.get_params(x[-1], (self.size, self.size))
        x = x[..., top:top+h, left:left+w]
        y = y[..., top:top+h, left:left+w]

        # Horizontal flip — negate wind_u (E↔W)
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
            x = x.clone()
            x[:, _WIND_U] = -x[:, _WIND_U]

        # Vertical flip — negate wind_v (N↔S)
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-2])
            y = torch.flip(y, dims=[-2])
            x = x.clone()
            x[:, _WIND_V] = -x[:, _WIND_V]

        # Random 90° rotation — rotate the wind vector accordingly
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k, dims=[-2, -1])
            y = torch.rot90(y, k, dims=[-2, -1])
            x = _rotate_wind(x, k)

        return x.contiguous(), y.contiguous()


class PairedCenterCrop:
    def __init__(self, size):
        from torchvision.transforms import CenterCrop
        self.crop = CenterCrop(size)

    def __call__(self, x, y):
        return self.crop(x), self.crop(y)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_iou(y_true, y_pred):
    preds = (y_pred > 0.5).float()
    intersection = (preds * y_true).sum()
    union = preds.sum() + y_true.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    bar = tqdm(loader, desc="Training (ConvLSTM v2)")
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        x = normalize_batch(x)
        optimizer.zero_grad()
        outputs = model(x)
        pred = outputs['spread_probability'] if isinstance(outputs, dict) else outputs
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
        bar.set_postfix(loss=f"{loss.item():.3f}", iou=f"{iou:.3f}")
    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = normalize_batch(x)
            outputs = model(x)
            pred = outputs['spread_probability'] if isinstance(outputs, dict) else outputs
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
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    args   = parser.parse_args()
    device = torch.device(args.device)
    print(f"🚀 Iniciando ConvLSTM v2 (solo Data Augmentation) en {device}")

    # Dataset already has 8 wind-corrected augmentations per sample baked in.
    # Using CenterCrop only — no extra runtime augmentation needed.
    train_ds = PatchDataset("data/processed/patches/spread_224/train",
                            transform=PairedCenterCrop(32))
    val_ds   = PatchDataset("data/processed/patches/spread_224/val",
                            transform=PairedCenterCrop(32))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    in_channels = len(CHANNELS) + 1  # 11 env + 1 fire mask
    model = RobustFireSpreadModel(
        input_channels=in_channels,
        hidden_dims=[64, 128],  # Idéntico al original
        dropout=0.15
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Parámetros: {params:,}  |  hidden_dims=[64, 128]")
    print(f"� Augmentaciones: HorizontalFlip + VerticalFlip + Rot90 activas")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalDiceLoss(alpha=0.9, gamma=2.0)

    best_iou = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n🏷️  Epoch {epoch+1}/{args.epochs}")
        t_loss, t_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"   Train Loss: {t_loss:.4f} | IoU: {t_iou:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   Val Loss:   {v_loss:.4f} | IoU: {v_iou:.4f}")
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), "models/best_convlstm_v2_spread.pth")
            print("   💾 ¡Modelo Guardado! (Mejor IoU)")

    print(f"\n✅ Finalizado. Mejor Val IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
