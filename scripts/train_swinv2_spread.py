import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Imports locales
from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)
from src.models.prop_swinv2 import SwinV2_3D_FirePrediction, DiceLoss, FocalLoss
from sklearn.metrics import accuracy_score, f1_score

def calculate_iou(y_true, y_pred):
    preds = (y_pred > 0.5).float()
    intersection = (preds * y_true).sum()
    union = preds.sum() + y_true.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# --- NORMALIZACIÓN (stats centralizadas en data/processed/spread_stats.json) ---
STATS = load_default_stats()

def normalize_batch(x):
    """Normaliza batch (B, T, C, H, W) usando stats centralizadas."""
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, smooth=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, reduction='mean')
        self.dice = DiceLoss(smooth)

    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.dice(inputs, targets)


class PhysicsInformedLoss(nn.Module):
    """
    Penalizes fire predictions that are spatially disconnected from Fire_Mask_T0.
    For each predicted pixel, if it is outside the morphologically dilated fire 
    reachability mask, it gets an extra penalty proportional to how confident 
    the prediction was.
    """
    def __init__(self, dilation_radius=5, lambda_physics=0.5):
        super().__init__()
        self.dilation_radius = dilation_radius
        self.lambda_physics = lambda_physics
        kernel_size = 2 * dilation_radius + 1
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size, kernel_size))

    def forward(self, pred_logits, fire_mask_t0):
        """
        pred_logits:  (B, 1, H, W) raw logits from the model
        fire_mask_t0: (B, 1, H, W) binary fire mask at time 0 (values 0 or 1)
        """
        # Dilate the fire mask to create a "reachable" zone
        fire_float = fire_mask_t0.float()
        k = self.kernel
        pad = self.dilation_radius
        # max_pool2d dilation
        reachable = F.max_pool2d(fire_float, kernel_size=k.shape[-1], stride=1, padding=pad)
        # Pixels outside the reachable zone
        outside_mask = (1.0 - reachable)  # 1 where fire CAN NOT reach
        
        # Probability of fire predicted at impossible pixels
        pred_prob = torch.sigmoid(pred_logits)
        impossible_fire = pred_prob * outside_mask
        
        return self.lambda_physics * impossible_fire.mean()


def topography_channel_dropout(x, drop_prob=0.3):
    """
    With probability drop_prob, zeros out slope_mean (idx 0) and elevation_mean (idx 1).
    Forces the model to learn from non-topographic features like Fire_Mask_T0.
    x shape: (B, T, C, H, W)
    """
    if random.random() < drop_prob:
        x = x.clone()
        # Indices 0 = elevation_mean, 1 = slope_mean based on CHANNELS list
        x[:, :, 0, :, :] = 0.0  # elevation_mean
        x[:, :, 1, :, :] = 0.0  # slope_mean
    return x

def train_epoch(model, loader, criterion, physics_loss_fn, optimizer, device, accum_steps=1):
    model.train()
    total_loss, total_iou = 0, 0
    progress_bar = tqdm(loader, desc="Training (SwinV2 3D)")
    
    optimizer.zero_grad()
    for i, (x, y) in enumerate(progress_bar):
        # x: (B, T, C, H, W). y: (B, 1, H, W)
        x, y = x.to(device), y.to(device)
        x = normalize_batch(x)
        
        # === TOPOGRAPHY CHANNEL DROPOUT ===
        # Randomly zero out elevation (ch 0) and slope (ch 1) with 30% probability.
        # Forces the model away from the topographic shortcut.
        x = topography_channel_dropout(x, drop_prob=0.3)
        
        # Extract fire mask from the last channel for physics loss
        fire_mask_t0 = x[:, 0, -1:, :, :]  # (B, 1, H, W) - first timestep, fire channel
        
        # Swin V2 Output: (B, T_out, 1, H, W)
        outputs = model(x)
        outputs_2d = torch.max(outputs, dim=1)[0]  # (B, 1, H, W)
        
        # === BASE LOSS ===
        base_loss = criterion(outputs_2d, y)
        
        # === PHYSICS-INFORMED LOSS ===
        # Penalizes predictions outside the fire reachability zone
        phys_loss = physics_loss_fn(outputs_2d, fire_mask_t0)
        
        loss = (base_loss + phys_loss) / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()
        
        scaled_loss = loss.item() * accum_steps
        total_loss += scaled_loss
        
        with torch.no_grad():
            probs = torch.sigmoid(outputs_2d)
            iou = calculate_iou(y, probs)
            total_iou += iou
            
        progress_bar.set_postfix(loss=f"{scaled_loss:.3f}", iou=f"{iou:.3f}")
        
    if device.type == "mps":
        torch.mps.empty_cache()
    
    return total_loss / len(loader), total_iou / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = normalize_batch(x)
            
            # Apply channel dropout in val too, to match train distribution
            x = topography_channel_dropout(x, drop_prob=0.3)
            
            outputs = model(x)
            outputs_2d = torch.max(outputs, dim=1)[0]
                
            loss = criterion(outputs_2d, y)
            total_loss += loss.item()
            total_iou += calculate_iou(y, torch.sigmoid(outputs_2d))
            
    return total_loss / len(loader), total_iou / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50) # Swin V2 needs more epochs to stabilize with augmentations
    parser.add_argument("--batch_size", type=int, default=1) # Reduced strictly to 1 to avoid MPS Out Of Memory
    parser.add_argument("--accum_steps", type=int, default=4) # Accumulate to simulate batch_size = 4
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"🚀 Iniciando entrenamiento SwinV2 3D en {device}")

    # Dataset Setup
    train_dir = "data/processed/patches/spread_224/train"
    val_dir = "data/processed/patches/spread_224/val"
    
    from torchvision.transforms import CenterCrop, RandomCrop
    import random
    class PairedRandomCrop:
        def __init__(self, size):
            self.size, self.crop = size, RandomCrop(size)
        def __call__(self, x, y):
            top, left, h, w = RandomCrop.get_params(x[-1], (self.size, self.size))
            x_crop = x[..., top:top+h, left:left+w]
            y_crop = y[..., top:top+h, left:left+w]
            
            # Spatial Augmentations
            if random.random() > 0.5:
                # Random Horizontal Flip
                x_crop = torch.flip(x_crop, dims=[-1])
                y_crop = torch.flip(y_crop, dims=[-1])
                
            if random.random() > 0.5:
                # Random Vertical Flip
                x_crop = torch.flip(x_crop, dims=[-2])
                y_crop = torch.flip(y_crop, dims=[-2])
                
            # Random 90 degree Rotations (0, 1, 2, or 3 times)
            k = random.randint(0, 3)
            if k > 0:
                x_crop = torch.rot90(x_crop, k, dims=[-2, -1])
                y_crop = torch.rot90(y_crop, k, dims=[-2, -1])
                
            return x_crop.contiguous(), y_crop.contiguous()
            
    class PairedCenterCrop:
        def __init__(self, size):
            self.crop = CenterCrop(size)
        def __call__(self, x, y):
            return self.crop(x), self.crop(y)

    train_ds = PatchDataset(train_dir, transform=PairedRandomCrop(32))
    val_ds = PatchDataset(val_dir, transform=PairedCenterCrop(32))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model Setup (In_chans = 11 features + 1 mask = 12)
    in_channels = len(CHANNELS) + 1 
    model = SwinV2_3D_FirePrediction(
        in_chans=in_channels,
        embed_dim=48, # Scale down for memory (vs 96)
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4), # Window temporal/spatial pooling
        window_size=(4, 4, 4),
        drop_rate=0.2,        # Added Dropout to prevent overfitting
        attn_drop_rate=0.1    # Added Attention Dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = FocalDiceLoss(alpha=0.9, gamma=2.0)
    physics_loss_fn = PhysicsInformedLoss(dilation_radius=5, lambda_physics=0.1).to(device)
    
    best_iou = 0.0
    Path("models").mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n🏷️  Epoch {epoch+1}/{args.epochs}")
        t_loss, t_iou = train_epoch(model, train_loader, criterion, physics_loss_fn, optimizer, device, args.accum_steps)
        v_loss, v_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"   Train Loss: {t_loss:.4f} | IoU: {t_iou:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"   Val Loss:   {v_loss:.4f} | IoU: {v_iou:.4f}")
        
        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), "models/best_swinv2_spread.pth")
            print("   💾 ¡Modelo Guardado! (Mejor IoU)")

if __name__ == "__main__":
    main()
