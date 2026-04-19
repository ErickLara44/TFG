
import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.models.prop import RobustFireSpreadModel
from scripts.visualize_spread import CHANNELS, normalize_batch

def inspect_sample_0():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🕵️‍♂️ Inspeccionando Muestra 0 de TEST en {device}")
    
    # 1. Load Data
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    
    # En visualize_spread.py usamos valid_indices.
    # Necesitamos saber cuál es el "Sample 0" de la visualización.
    # El script iteraba sobre valid_indices. El archivo se llama evolution_0.png,
    # asumo que corresponde al índice 0 del dataset original o al primero de valid_indices?
    # En el script visualize: `save_path = f"{save_dir}/evolution_{idx}.png"`
    # Por tanto, es el índice idx del dataset global.
    
    IDX_TO_INSPECT = 0
    
    x, y = ds[IDX_TO_INSPECT]
    x_tensor = x.unsqueeze(0).to(device)
    x_tensor = normalize_batch(x_tensor)
    
    # 2. Load Model
    in_channels = len(CHANNELS) + 1
    model = RobustFireSpreadModel(input_channels=in_channels).to(device)
    model.load_state_dict(torch.load("models/best_spread_model.pth", map_location=device))
    model.eval()
    
    # 3. Predict
    with torch.no_grad():
        outputs = model(x_tensor)
        if isinstance(outputs, dict):
            pred = outputs['spread_probability']
        else:
            pred = outputs
            
    pred_map = pred.squeeze().cpu().numpy()
    target_map = y.squeeze().cpu().numpy()
    
    print(f"\n📊 ANÁLISIS MUESTRA {IDX_TO_INSPECT}:")
    print(f"   Max Probabilidad Predicha: {pred_map.max():.4f}")
    print(f"   Mean Probabilidad Predicha: {pred_map.mean():.4f}")
    print(f"   Threshold usado en mapa: 0.5")
    
    # Ver si hay algún pixel por encima de 0.1, 0.2...
    print(f"   Pixels > 0.1: {(pred_map > 0.1).sum()}")
    print(f"   Pixels > 0.3: {(pred_map > 0.3).sum()}")
    print(f"   Pixels > 0.5: {(pred_map > 0.5).sum()} (Esto es lo que dibuja el cian)")
    
    if pred_map.max() < 0.5:
        print("\n💡 CONCLUSIÓN: El modelo no dibujó nada porque ninguna probabilidad superó el 50%.")
        print(f"   Estuvo cerca? Max={pred_map.max():.2f}")

if __name__ == "__main__":
    inspect_sample_0()
