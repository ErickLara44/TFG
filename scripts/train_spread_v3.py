"""
train_spread_v3.py — ConvLSTM con Direction Head + DirectionalConsistencyLoss + ROS
====================================================================================
Extiende v2 con:
1. Entrenamiento del `direction_head` (ya presente en RobustFireSpreadModel):
   - Predice vector (dx, dy) ≡ (cos θ, sin θ) → cono de propagación dominante.
2. DirectionalConsistencyLoss (de prop.py): alinea (dx,dy) con el gradiente
   espacial del frente de fuego observado.
3. Rate of Spread (ROS) calculado como post-proceso: 
   distancia medida del frente / días hasta la llegada.
4. Augmentaciones de viento físicamente correctas (de v2).

Pérdida total:
  L = FocalDiceLoss(spread_prob, y) + λ_dir * DirectionalConsistencyLoss(dir_pred, spread_prob, y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
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
from src.models.prop import RobustFireSpreadModel, DirectionalConsistencyLoss

# ─────────────────────────────────────────────────────────────────────────────
STATS = load_default_stats()
_WIND_U, _WIND_V = get_wind_indices(CHANNELS)

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

# ─────────────────────────────────────────────────────────────────────────────
# LOSSES
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
# WIND-CORRECT AUGMENTATIONS (from v2)
# ─────────────────────────────────────────────────────────────────────────────
def _rotate_wind(x, k):
    """Rotate wind vector (u, v) by k*90° CCW to match spatial rot90."""
    x = x.clone()
    u = x[:, _WIND_U].clone()
    v = x[:, _WIND_V].clone()
    if k == 1:
        x[:, _WIND_U] = -v; x[:, _WIND_V] = u
    elif k == 2:
        x[:, _WIND_U] = -u; x[:, _WIND_V] = -v
    elif k == 3:
        x[:, _WIND_U] = v;  x[:, _WIND_V] = -u
    return x


class PairedRandomCropAug:
    def __init__(self, size):
        self.size = size

    def __call__(self, x, y):
        from torchvision.transforms import RandomCrop
        top, left, h, w = RandomCrop.get_params(x[-1], (self.size, self.size))
        x = x[..., top:top+h, left:left+w]
        y = y[..., top:top+h, left:left+w]

        if random.random() > 0.5:
            x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1])
            x = x.clone(); x[:, _WIND_U] = -x[:, _WIND_U]

        if random.random() > 0.5:
            x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2])
            x = x.clone(); x[:, _WIND_V] = -x[:, _WIND_V]

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
# RATE OF SPREAD (Post-procesado)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ros(fire_mask_t0: torch.Tensor, fire_pred: torch.Tensor,
                cell_size_km: float = 1.0) -> float:
    """
    Calcula la Rate of Spread (ROS) en km/día.
    
    Asume predicción de 1 día hacia adelante.
    ROS = distancia media del frente nuevo / 1 día
    
    Args:
        fire_mask_t0: (B, 1, H, W) máscara de fuego en T=0 (binaria)
        fire_pred:    (B, 1, H, W) predicción de la siguiente máscara (probabilidad)
        cell_size_km: tamaño de celda en km (default=1km)
    
    Returns:
        ROS media en km/día para el batch
    """
    fire_pred_bin = (fire_pred > 0.5).float()
    
    # Celdas nuevas encendidas (no están en t0 pero sí en t1)
    new_fire = fire_pred_bin * (1 - fire_mask_t0.float())
    
    ros_batch = []
    for b in range(fire_mask_t0.shape[0]):
        mask_0 = fire_mask_t0[b, 0].cpu().numpy()
        new_f   = new_fire[b, 0].cpu().numpy()
        
        if mask_0.sum() == 0 or new_f.sum() == 0:
            continue  # No hay fuego o no hay propagación
        
        # Centro de masa del fuego inicial
        ys, xs = np.where(mask_0 > 0)
        cy0, cx0 = ys.mean(), xs.mean()
        
        # Centro de masa del frente nuevo
        yn, xn = np.where(new_f > 0)
        cy1, cx1 = yn.mean(), xn.mean()
        
        # Distancia euclidiana en celdas → km
        dist_cells = np.sqrt((cy1 - cy0)**2 + (cx1 - cx0)**2)
        dist_km = dist_cells * cell_size_km
        
        ros_batch.append(dist_km)  # En 1 día de predicción
    
    return float(np.mean(ros_batch)) if ros_batch else 0.0

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
def train_epoch(model, loader, criterion, dir_loss_fn, optimizer, device,
                lambda_dir=0.1):
    model.train()
    total_loss, total_iou = 0.0, 0.0
    bar = tqdm(loader, desc="Training (ConvLSTM v3)")

    for x, y in bar:
        x, y = x.to(device), y.to(device)
        x = normalize_batch(x)

        optimizer.zero_grad()
        outputs = model(x)  # dict with 'spread_probability', 'propagation_direction'

        pred  = outputs['spread_probability']    # (B, 1, H, W)
        d_vec = outputs['propagation_direction']  # (B, 2, H, W) — (cos θ, sin θ)

        if pred.shape != y.shape:
            pred = pred.view_as(y)

        # Spread loss (Focal + Dice)
        spread_loss = criterion(pred, y)
        
        # Directional consistency loss
        # Needs float y (not binary) for gradient computation
        d_loss = dir_loss_fn(d_vec, pred, y)

        loss = spread_loss + lambda_dir * d_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            iou = calculate_iou(y, pred)
            total_iou += iou

        bar.set_postfix(loss=f"{loss.item():.3f}", iou=f"{iou:.3f}")

    return total_loss / len(loader), total_iou / len(loader)


def validate(model, loader, criterion, dir_loss_fn, device, lambda_dir=0.1):
    model.eval()
    total_loss, total_iou, total_ros = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = normalize_batch(x)
            outputs = model(x)
            pred  = outputs['spread_probability']
            d_vec = outputs['propagation_direction']
            if pred.shape != y.shape:
                pred = pred.view_as(y)

            d_loss = dir_loss_fn(d_vec, pred, y)
            total_loss += (criterion(pred, y) + lambda_dir * d_loss).item()
            total_iou  += calculate_iou(y, pred)

            # ROS: extraemos fire mask en t0 del último canal de entrada
            fire_t0 = x[:, -1, -1:, :, :]  # last timestep, fire mask channel
            total_ros += compute_ros(fire_t0, pred)

    n = len(loader)
    return total_loss / n, total_iou / n, total_ros / n

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lambda_dir", type=float, default=0.1,
                        help="Weight for DirectionalConsistencyLoss")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    args   = parser.parse_args()
    device = torch.device(args.device)
    print(f"🚀 ConvLSTM v3 (Direction + ROS) en {device}")

    # Dataset already has 8 wind-corrected augmentations per sample baked in.
    # Using CenterCrop only — no extra runtime augmentation needed.
    train_ds = PatchDataset("data/processed/patches/spread_224/train",
                            transform=PairedCenterCrop(32))
    val_ds   = PatchDataset("data/processed/patches/spread_224/val",
                            transform=PairedCenterCrop(32))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    in_channels = len(CHANNELS) + 1  # 11 env + 1 fire mask
    model = RobustFireSpreadModel(
        input_channels=in_channels,
        hidden_dims=[64, 128],
        dropout=0.15
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Parámetros: {params:,}")
    print(f"⚖️  FocalDiceLoss + λ={args.lambda_dir}·DirectionalConsistencyLoss")
    print(f"🎲  Augmentaciones con wind-correction activas")
    print(f"📐  ROS calculado en cada época de validación")

    optimizer     = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=args.epochs,
                                                          eta_min=1e-6)
    criterion     = FocalDiceLoss(alpha=0.9, gamma=2.0)
    dir_loss_fn   = DirectionalConsistencyLoss()

    best_iou = 0.0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n🏷️  Epoch {epoch+1}/{args.epochs}")
        t_loss, t_iou = train_epoch(model, train_loader, criterion, dir_loss_fn,
                                    optimizer, device, args.lambda_dir)
        v_loss, v_iou, v_ros = validate(model, val_loader, criterion, dir_loss_fn,
                                        device, args.lambda_dir)
        scheduler.step()
        print(f"   Train  Loss: {t_loss:.4f} | IoU: {t_iou:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   Val    Loss: {v_loss:.4f} | IoU: {v_iou:.4f}")
        print(f"   Val ROS:     {v_ros:.4f} km/día")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), "models/best_convlstm_v3_spread.pth")
            print("   💾 ¡Modelo Guardado! (Mejor IoU)")

    print(f"\n✅ Finalizado. Mejor Val IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()
