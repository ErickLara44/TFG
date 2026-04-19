
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel

STATS = load_default_stats()

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

def visualize_results():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Visualizando en {device} - DATASET DE TEST")
    
    # 1. Load Data (NO CROP for visualization map)
    # Usamos el conjunto de TEST para evaluar la realidad final
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir) # No transforms -> 224x224
    
    # Filtrar solo muestras con fuego significativo para que la visualización valga la pena
    print("🔍 Buscando muestras con fuego activo en TEST...")
    valid_indices = []
    for i in range(len(ds)):
        try:
            x, y = ds[i]
            # Check if there is fire in input (last channel of last timestep)
            # x shape: (T, C, H, W) -> C=12 (11 feats + 1 mask)
            fire_input = x[-1, -1] 
            if fire_input.sum() > 5: # Al menos 5 pixels de fuego activo (más permisivo en test)
                valid_indices.append(i)
        except:
            pass
            
    print(f"✅ Encontradas {len(valid_indices)} muestras con fuego activo en test.")
    if not valid_indices:
        print("❌ No se encontraron muestras con suficiente fuego en test.")
        return

    # 2. Load Model
    in_channels = len(CHANNELS) + 1
    model = RobustFireSpreadModel(input_channels=in_channels, hidden_dims=[64, 128]).to(device)
    
    checkpoint_path = "models/best_spread_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Modelo cargado.")
    else:
        print("❌ No se encontró el checkpoint.")
        return
        
    model.eval()
    
    # 3. Predict & Plot - TODOS LOS CASOS DE TEST
    # En test queremos ver TODO lo que tenga fuego
    indices_to_plot = valid_indices
    
    save_dir = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/outputs_spread"
    os.makedirs(save_dir, exist_ok=True)
    
    from matplotlib.colors import ListedColormap
    
    print(f"📸 Generando mapas para {len(indices_to_plot)} casos de test...")
    
    for idx_i, idx in enumerate(tqdm(indices_to_plot)):
        x, y = ds[idx]
        x_tensor = x.unsqueeze(0).to(device)
        x_tensor = normalize_batch(x_tensor)
        
        with torch.no_grad():
            outputs = model(x_tensor)
            if isinstance(outputs, dict):
                pred = outputs['spread_probability']
            else:
                pred = outputs
                
        # Numpy conversion
        # t=0 (Input Fire): Último timestep, canal mask
        fire_t0 = x[-1, -1].numpy() 
        # t+1 (Target Fire)
        fire_t1_true = y.squeeze().numpy()
        # t+1 (Pred Fire)
        fire_t1_pred = pred.squeeze().cpu().numpy()
        
        # --- CREAR MAPA DE CONFUSIÓN DE LA PROPAGACIÓN ---
        # Umbral de decisión
        thr = 0.5
        
        # 1. Background: Elevación
        plt.figure(figsize=(10, 10))
        elevation = x[-1, 0].numpy()
        plt.imshow(elevation, cmap='gray', alpha=0.4)
        
        # 2. Fuego Inicial (Contexto: dondé empezó la simulación)
        mask_t0 = np.ma.masked_where(fire_t0 < 0.5, fire_t0)
        plt.imshow(mask_t0, cmap=ListedColormap(['#800080']), alpha=0.4, label='Fuego t=0') # Morado suave
        
        # 3. Matrices de confusión (Basado en la Realidad t1 vs Predicción t1)
        true_positive = (fire_t1_true > 0.5) & (fire_t1_pred >= thr)
        false_negative = (fire_t1_true > 0.5) & (fire_t1_pred < thr) # Fuego real no detectado
        
        # 3.1 Falsos Negativos (Fuego real que nos perdimos) -> ROJO/NARANJA
        mask_fn = np.ma.masked_where(false_negative == 0, false_negative)
        plt.imshow(mask_fn, cmap=ListedColormap(['red']), alpha=0.8, label='FN (No detectado)')
        
        # 3.2 True Positives (Fuego real detectado) -> VERDE
        mask_tp = np.ma.masked_where(true_positive == 0, true_positive)
        plt.imshow(mask_tp, cmap=ListedColormap(['lime']), alpha=1.0, label='TP (Acierto)')
        
        # 3.3 Falsos Positivos / Probabilidad (Predicción sin fuego real) -> CIAN/HEATMAP
        # Mostramos la probabilidad continua para ver el "nivel de falsas alarmas" o riesgo
        fp_mask = (fire_t1_true < 0.5) & (fire_t1_pred > 0.1) # Probabilidad > 10% en zonas sin fuego
        prob_fp = np.where(fp_mask, fire_t1_pred, 0)
        mask_fp_heat = np.ma.masked_where(prob_fp == 0, prob_fp)
        
        # Colormap cool (Cyan->Azul->Magenta) para que no se confunda con el rojo/verde
        img_pred = plt.imshow(mask_fp_heat, cmap='cool', alpha=0.6, vmin=0, vmax=1)
        plt.colorbar(img_pred, fraction=0.046, pad=0.04, label="Probabilidad de Falsas Alarmas (FP)") 
        
        # Leyenda manual
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        custom_lines = [
            Line2D([0], [0], color='#800080', lw=4, alpha=0.4, label='Origen: Fuego en t=0'),
            Line2D([0], [0], color='lime', lw=4, label='TP: Acierto (Predicho + Real)'),
            Line2D([0], [0], color='red', lw=4, label='FN: Fuego REAL ignorado'),
            mpatches.Patch(color='cyan', alpha=0.6, label='FP: Falsa Alarma (Prob > 0.1)')
        ]
        plt.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.6, 1))
        
        plt.title(f"Dinámica vs Predicción: Muestra {idx}")
        plt.axis('off')
        
        save_path = f"{save_dir}/dynamics_{idx}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    visualize_results()
