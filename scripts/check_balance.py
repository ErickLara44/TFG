
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset_patch import PatchDataset

def check_balance():
    train_dir = "data/processed/patches/spread_224/train"
    
# 1. Definir Crop 16x16 Global para evitar errores de pickle
crop = CenterCrop(16)

class CropTransform:
    def __call__(self, x, y):
        return crop(x), crop(y)

def check_balance():
    train_dir = "data/processed/patches/spread_224/train"
    
    ds = PatchDataset(train_dir, transform=CropTransform())
    # num_workers=0 para evitar problemas de multiprocessing en scripts simples
    loader = DataLoader(ds, batch_size=32, num_workers=0)
    
    print(f"📊 Analizando balance de clases en {len(ds)} muestras (Crop 16x16)...")
    
    total_pixels = 0
    total_fire = 0
    
    for _, y in tqdm(loader):
        # y shape: (B, 1, 16, 16)
        total_pixels += y.numel()
        total_fire += y.sum().item()
        
    fire_ratio = total_fire / total_pixels
    non_fire_ratio = 1 - fire_ratio
    pos_weight = non_fire_ratio / (fire_ratio + 1e-6)
    
    print("\n⚖️ RESULTADOS (16x16 km):")
    print(f"   Total Píxeles: {total_pixels}")
    print(f"   Píxeles Fuego: {total_fire:.0f} ({fire_ratio*100:.2f}%)")
    print(f"   Píxeles Sin Fuego: {total_pixels - total_fire:.0f} ({non_fire_ratio*100:.2f}%)")
    print(f"   Balance (Neg/Pos): {pos_weight:.1f} : 1")
    
    # Comparativa teórica con 224x224
    # Si asumimos que el fuego está centrado, en 224x224 el % sería mucho menor
    # (16*16) / (224*224) = 0.005 -> La señal se diluye 200 veces.

if __name__ == "__main__":
    check_balance()
