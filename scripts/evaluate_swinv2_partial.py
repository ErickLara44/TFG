import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

# Import necessary modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.prop_swinv2 import SwinV2_3D_FirePrediction
from src.data.dataset_patch import PatchDataset

# Metrics
def calculate_iou(pred, target, threshold=0.5):
    # Flatten everything past batch dimension to handle B,T,H,W vs B,T,C,H,W mismatch
    pred_mask = (pred > threshold).float().view(pred.shape[0], -1)
    target_mask = (target > threshold).float().view(target.shape[0], -1)
    
    intersection = (pred_mask * target_mask).sum(dim=1)
    union = pred_mask.sum(dim=1) + target_mask.sum(dim=1) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")

    # Load dataset
    test_dir = "data/processed/patches/spread_224/test"
    if not os.path.exists(test_dir):
        logger.error(f"Test directory no encontrado: {test_dir}")
        return

    test_dataset = PatchDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Load Model
    model = SwinV2_3D_FirePrediction(
        in_chans=12,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4),
        window_size=(4, 4, 4)
    ).to(device)

    # Load checkpoint
    ckpt_path = "models/epoch1_physics_informed.pth"
    if not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint no encontrado: {ckpt_path}. ¡El entrenamiento aún no ha guardado el primer mejor modelo!")
        return

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.success(f"Checkpoint cargado: {ckpt_path}")
    except Exception as e:
        logger.error(f"Error cargando checkpoint (puede que se esté escribiendo ahora mismo): {e}")
        return

    # Evaluate
    model.eval()
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing Partial Model"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Normalization (same as training)
            inputs[:, :, :-1] = (inputs[:, :, :-1] - inputs[:, :, :-1].mean(dim=(1,3,4), keepdim=True)) / \
                                (inputs[:, :, :-1].std(dim=(1,3,4), keepdim=True) + 1e-6)
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            
            iou = calculate_iou(preds, targets)
            total_iou += iou
            num_batches += 1

    avg_iou = total_iou / num_batches
    logger.info(f"\n=========================================")
    logger.info(f"📊 Final Test IoU ({num_batches} batches): {avg_iou:.4f}")
    logger.info(f"=========================================\n")

if __name__ == '__main__':
    main()
