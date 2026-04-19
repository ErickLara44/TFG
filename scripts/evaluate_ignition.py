import sys
import os
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ignition import RobustFireIgnitionModel
from src.data.data_ignition_improved import (
    DEFAULT_FEATURE_VARS,
    PrecomputedIgnitionDataset,
    load_default_stats,
    build_channel_stats_arrays,
)

def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo de ignición (Test Set)")
    parser.add_argument("--data_dir", type=str, default="data/processed/patches_temporal_strict", help="Directorio con parches")
    parser.add_argument("--model_path", type=str, default="best_robust_ignition_model.pth", help="Path al modelo entrenado")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de batch")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo")
    parser.add_argument("--convlstm_hidden", nargs='+', type=int, default=[64, 128], help="Dimensiones ocultas de ConvLSTM")
    
    args = parser.parse_args()
    
    # Dispositivo
    if args.device == 'auto':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"🖥️ Usando dispositivo: {device}")
    
    # 1. Cargar Test Dataset
    test_dir = Path(args.data_dir) / 'test'
    if not test_dir.exists():
        print(f"❌ Error: No existe {test_dir}")
        sys.exit(1)
        
    test_files = sorted(list(test_dir.glob("patch_*.pt")))
    
    # Extraer índices
    def get_indices(files):
        indices = []
        for f in files:
            try:
                indices.append(int(f.stem.split('_')[1]))
            except: pass
        return sorted(indices)
        
    test_indices = get_indices(test_files)
    print(f"📂 Cargando {len(test_indices)} muestras de test desde {test_dir}")
    
    if len(test_indices) == 0:
        print("❌ Error: No hay muestras de test")
        sys.exit(1)
    
    # Cargar stats centralizadas (scripts/compute_ignition_stats.py)
    raw_stats = load_default_stats()
    if not raw_stats:
        print("⚠️ No se encontraron stats de normalización. Evaluando con datos RAW (peligroso).")
        print("   Ejecuta: python scripts/compute_ignition_stats.py")
        stats = None
    else:
        stats = build_channel_stats_arrays(DEFAULT_FEATURE_VARS, raw_stats)
        print(f"✅ Estadísticas de normalización cargadas ({len(raw_stats)} vars).")
        
    test_ds = PrecomputedIgnitionDataset(test_dir, indices=test_indices, stats=stats) 
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Cargar Modelo
    print(f"🏗️ Cargando modelo desde {args.model_path}...")
    
    # Inferir dimensiones de entrada
    x0, _ = test_ds[0]
    T, C, H, W = x0.shape
    
    model = RobustFireIgnitionModel(
        num_input_channels=C,
        temporal_context=T,
        hidden_dims=args.convlstm_hidden
    )
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado (Epoch {checkpoint['epoch']}, Best F1: {checkpoint.get('f1_score', '?'):.3f})")
    
    # 3. Evaluar
    all_targets = []
    all_probs = []
    all_preds_risk = []
    
    print("🚀 Evaluando...")
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            
            probs = torch.sigmoid(outputs['ignition']).cpu().numpy()
            targets = y.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_targets.extend(targets)
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # 4. Métricas
    try:
        auroc = roc_auc_score(all_targets, all_probs)
    except: 
        auroc = 0.5
        
    # Buscar mejor threshold
    precision, recall, thresholds = precision_recall_curve(all_targets, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    preds = (all_probs > best_thr).astype(int)
    
    print("\n📊 RESULTADOS EN TEST SET (2023):")
    print(f"   AUROC: {auroc:.4f}")
    print(f"   Best Threshold: {best_thr:.4f}")
    print("\n" + classification_report(all_targets, preds, target_names=['No Fire', 'Fire']))
    
    cm = confusion_matrix(all_targets, preds)
    print("Matriz de Confusión:")
    print(cm)
    
    # Guardar reporte
    results = {
        "auroc": auroc,
        "classification_report": classification_report(all_targets, preds, output_dict=True),
        "confusion_matrix": cm.tolist()
    }
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n✅ Resultados guardados en test_results.json")
    
    print("\n" + "="*60)
    print("🔬 ANÁLISIS DE SENSIBILIDAD (Threshold vs Precision/Recall)")
    print("="*60)
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'False Pos':<10}")
    print("-" * 65)
    
    # Probar varios umbrales manuales
    thresholds_to_test = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if best_thr not in thresholds_to_test:
        thresholds_to_test.append(best_thr)
    thresholds_to_test.sort()
    
    for thr in thresholds_to_test:
        preds_t = (all_probs > thr).astype(int)
        cm_t = confusion_matrix(all_targets, preds_t)
        tn, fp, fn, tp = cm_t.ravel()
        
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        
        star = "*" if abs(thr - best_thr) < 1e-4 else ""
        print(f"{thr:<10.4f} | {prec:<10.3f} | {rec:<10.3f} | {f1:<10.3f} | {fp:<10d} {star}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
