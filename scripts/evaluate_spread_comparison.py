import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel
from src.models.prop_swinv2 import SwinV2_3D_FirePrediction
from torchvision.transforms import CenterCrop

# Métricas
def calculate_iou(y_true, y_pred):
    preds = (y_pred > 0.5).float()
    intersection = (preds * y_true).sum()
    union = preds.sum() + y_true.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

STATS = load_default_stats()

def normalize_batch(x, device):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=device
    )
    return (x - mean_t) / std_t

class PairedCenterCrop:
    def __init__(self, size):
        self.crop = CenterCrop(size)
    def __call__(self, x, y):
        return self.crop(x), self.crop(y)

def evaluate_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluando en {device}...")
    
    test_dir = "data/processed/patches/spread_224/test"
    test_ds = PatchDataset(test_dir, transform=PairedCenterCrop(32))
    val_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    # 1. Cargar Original (RobustFireSpreadModel)
    in_channels = len(CHANNELS) + 1 
    model_orig = RobustFireSpreadModel(input_channels=in_channels, hidden_dims=[64, 128]).to(device)
    model_orig.load_state_dict(torch.load("models/best_spread_model.pth", map_location=device, weights_only=True))
    model_orig.eval()
    
    # 2. Cargar SwinV2
    model_swin = SwinV2_3D_FirePrediction(
        in_chans=in_channels, embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4), window_size=(4, 4, 4)
    ).to(device)
    model_swin.load_state_dict(torch.load("models/best_swinv2_spread.pth", map_location=device, weights_only=True))
    model_swin.eval()
    
    iou_orig_total, iou_swin_total = 0, 0
    batches = 0
    
    with torch.no_grad():
        # Vamos a evaluar solo los primeros 50 batches para que sea rápido (200 muestras)
        # o todo si es rápido
        for x, y in tqdm(val_loader, desc="Testing both models"):
            x, y = x.to(device), y.to(device)
            x_norm = normalize_batch(x, device)
            
            # Predicción Original
            out_orig = model_orig(x_norm)
            if isinstance(out_orig, dict): out_orig = out_orig['spread_probability']
            if out_orig.shape != y.shape: out_orig = out_orig.view_as(y)
            iou_orig = calculate_iou(y, out_orig)
            iou_orig_total += iou_orig.item()
            
            # Predicción Swin
            out_swin = model_swin(x_norm)
            out_swin_2d = torch.max(out_swin, dim=1)[0]
            iou_swin = calculate_iou(y, torch.sigmoid(out_swin_2d))
            iou_swin_total += iou_swin.item()
            
            batches += 1
                
    print(f"\n--- RESULTADOS VALIDACIÓN ({batches * 4} muestras) ---")
    print(f"Modelo Original (ConvLSTM) IoU: {iou_orig_total / batches:.4f}")
    print(f"Modelo Swin V2 3D          IoU: {iou_swin_total / batches:.4f}")

if __name__ == '__main__':
    evaluate_models()
