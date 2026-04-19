
import torch
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset

def visualize_truth_jump():
    print("🕵️‍♂️ Visualizando VERDAD DEL TERRENO (Sin Modelo)...")
    
    # Cargar dataset de test
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    
    # Caso conocido de "Salto": Muestra 21 (según auditoría anterior)
    idx = 21
    
    x, y = ds[idx]
    
    # Extraer máscaras de fuego REALES
    fire_t0 = x[-1, -1].numpy()  # Input (t)
    fire_t1 = y.squeeze().numpy() # Target (t+1)
    
    # Verificar superposición
    overlap = ((fire_t0 > 0.5) & (fire_t1 > 0.5)).sum()
    
    print(f"Muestra {idx}:")
    print(f"- Píxeles Fuego t=0: {(fire_t0 > 0.5).sum()}")
    print(f"- Píxeles Fuego t=1: {(fire_t1 > 0.5).sum()}")
    print(f"- Superposición: {overlap}")
    
    # --- PLOT ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Fuego Inicial (t)
    axes[0].imshow(fire_t0, cmap='Reds', vmin=0, vmax=1)
    axes[0].set_title(f"INPUT REAL (t)\n{(fire_t0 > 0.5).sum()} px")
    
    # 2. Fuego Final (t+1)
    axes[1].imshow(fire_t1, cmap='Oranges', vmin=0, vmax=1)
    axes[1].set_title(f"TARGET REAL (t+1)\n{(fire_t1 > 0.5).sum()} px")
    
    # 3. Superposición
    # Crear imagen combinada
    combined = np.zeros((224, 224, 3))
    # Rojo para t
    combined[(fire_t0 > 0.5)] = [1, 0, 0] 
    # Verde para t+1 (para que se vea bien el contraste)
    combined[(fire_t1 > 0.5)] = [0, 1, 0]
    # Amarillo para overlap (Rojo + Verde)
    combined[(fire_t0 > 0.5) & (fire_t1 > 0.5)] = [1, 1, 0]
    
    axes[2].imshow(combined)
    axes[2].set_title(f"SUPERPOSICIÓN REAL\n(Rojo=Ayer, Verde=Hoy)")
    
    save_dir = "outputs/debug"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/truth_jump_{idx}.png"
    plt.savefig(save_path)
    plt.close()
    
    print(f"📸 Evidencia guardada en: {save_path}")

if __name__ == "__main__":
    visualize_truth_jump()
