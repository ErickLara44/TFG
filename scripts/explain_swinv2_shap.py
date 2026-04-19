import sys
import os
import torch
import shap
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.prop_swinv2 import SwinV2_3D_FirePrediction
from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)

FEATURE_VARS = list(CHANNELS) + ['Fire_Mask_T0']
STATS = load_default_stats()

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / (std_t + 1e-6)

def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for Swin V2 Fire Spread Model")
    parser.add_argument("--data_dir", type=str, default="data/processed/patches/spread_224/test")
    parser.add_argument("--model_path", type=str, default="models/best_swinv2_spread.pth")
    parser.add_argument("--crop_size", type=int, default=32, help="Tamaño de recorte (Center Crop) matching training")
    parser.add_argument("--background_size", type=int, default=15, help="Tamaño dataset fondo SHAP")
    parser.add_argument("--test_size", type=int, default=5, help="Muestras a explicar")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"🖥️ Usando dispositivo para SHAP: {device}")

    # 1. Dataset
    print(f"📂 Cargando parches desde {args.data_dir}...")
    dataset = PatchDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("❌ Dataset vacío. Revisa la ruta.")
        return

    loader = DataLoader(dataset, batch_size=args.background_size + args.test_size, shuffle=True)
    try:
        batch_x, batch_y = next(iter(loader))
    except StopIteration:
        print("❌ No hay suficientes datos.")
        return

    batch_x = normalize_batch(batch_x).to(device)
    
    # On the fly Center Crop to emulate training input
    import torchvision.transforms.functional as TF
    if args.crop_size < 224:
        B, T, C, H, W = batch_x.shape
        x_crops = []
        for t in range(T):
            x_t = batch_x[:, t] # (B, C, H, W)
            x_t_cropped = TF.center_crop(x_t, output_size=args.crop_size)
            x_crops.append(x_t_cropped.unsqueeze(1))
        batch_x = torch.cat(x_crops, dim=1) # (B, T, C, crop_H, crop_W)

    # 2. Modelo
    model = SwinV2_3D_FirePrediction(
        in_chans=12,
        embed_dim=48, 
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4), 
        window_size=(4, 4, 4)
    ).to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("✅ Weights loaded.")
    else:
        print(f"⚠️ CUIDADO: Weights no encontrados en {args.model_path}")
    
    model.eval()

    # SHAP DeepExplainer no soporta ReLU(inplace=True) ni algunas funciones inplace
    print("🔧 Ajustando modelo para SHAP (RELU inplace=False)...")
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
        if isinstance(module, torch.nn.GELU):
            pass # GELU is fine

    # 3. Preparar Background y Test
    background = batch_x[:args.background_size]
    test_samples = batch_x[args.background_size:]
    
    # Wrapper para SHAP - reduce a escalar (pool global temporal y espacial)
    class SwinWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            out = self.model(x)
            # Output is (B, T_out, 1, H, W)
            # Reducir todo a un escalar promedio para que DeepExplainer o GradientExplainer
            # no tengan que manejar gradientes múltiples dimensionales
            return out.mean(dim=[1, 2, 3, 4]).view(x.size(0), -1)

    wrapped_model = SwinWrapper(model)
    
    # GradientExplainer suele ser más compatible con arquitecturas complejas (Transformers 3D) que DeepExplainer
    print("🧠 Iniciando SHAP GradientExplainer (CPU)...")
    try:
        explainer = shap.GradientExplainer(wrapped_model, background)
        
        print("💥 Calculando valores SHAP iterativamente...")
        shap_values_list = []
        
        for i in range(test_samples.shape[0]):
            print(f"   Procesando muestra {i+1}/{test_samples.shape[0]}...")
            s_val = explainer.shap_values(test_samples[i:i+1])
            
            if isinstance(s_val, list):
                s_val = s_val[0]
                
            if isinstance(s_val, list):
                s_val = np.mean(np.array(s_val), axis=0)
                
            if isinstance(s_val, torch.Tensor):
                s_val = s_val.detach().cpu().numpy()
                
            shap_values_list.append(s_val)

        shap_values = np.concatenate(shap_values_list, axis=0)
        
        if shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]
            
        print(f"✅ SHAP Values shape final: {shap_values.shape}")
        
    except Exception as e:
        print(f"❌ Error con GradientExplainer computando SwinV2: {e}")
        return

    # --- 1. BAR PLOT ---
    print("📊 Generando Bar Plot...")
    # shape: (B, T, C, H, W). Promediamos todo menos el canal (C)
    shap_values_bar = np.abs(shap_values).mean(axis=(0, 1, 3, 4))
    shap_values_bar = np.array(shap_values_bar).flatten()
    
    # Print numerical values
    print("\n--- SHAP Feature Importances (Swin V2 3D) ---")
    sorted_idx = np.argsort(shap_values_bar)[::-1]
    for i in sorted_idx:
        print(f"{FEATURE_VARS[i]}: {shap_values_bar[i]:.6f}")
    print("-------------------------------------------\n")
    
    plt.figure()
    indices = np.argsort(shap_values_bar)
    plt.barh(range(len(indices)), shap_values_bar[indices], color='#8855ff')
    plt.yticks(range(len(indices)), [FEATURE_VARS[i] for i in indices])
    plt.xlabel("mean(|SHAP value|) - Swin V2")
    plt.title("Swin V2 3D Feature Importance")
    plt.tight_layout()
    plt.savefig("shap_swinv2_importance_bar.png")
    
    # --- 2. BEESWARM PLOT ---
    print("📊 Generando Beeswarm Plot...")
    # aplanamos las dimensiones para comparar contribuciones vs feature real
    shap_t = shap_values.transpose(0, 1, 3, 4, 2).reshape(-1, len(FEATURE_VARS))
    feat_t = test_samples.permute(0, 1, 3, 4, 2).reshape(-1, len(FEATURE_VARS)).cpu().numpy()
    
    num_points = shap_t.shape[0]
    max_display = 2000
    if num_points > max_display:
        idx = np.random.choice(num_points, max_display, replace=False)
        shap_subset = shap_t[idx]
        feat_subset = feat_t[idx]
    else:
        shap_subset = shap_t
        feat_subset = feat_t
        
    plt.figure()
    shap.summary_plot(shap_subset, feat_subset, feature_names=FEATURE_VARS, show=False)
    plt.tight_layout()
    plt.savefig("shap_swinv2_beeswarm.png")
    print("✅ Todo completado. Gráficos guardados.")

if __name__ == "__main__":
    main()
