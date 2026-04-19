
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.data_tab import SpainCubeFireDataset

def calculate_fire_metrics(y_true, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auroc = roc_auc_score(y_true, y_pred_proba)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auroc': auroc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    print("🔬 ABLATION STUDY: Entrenando XGBoost SIN 'is_near_fire'")
    print("=======================================================")

    # 1. Cargar datos
    print("📦 Cargando datos...")
    ds = SpainCubeFireDataset("dummy.nc", output_dir="data/processed")
    
    X_train, y_train, _ = ds.get_features_labels("train")
    X_val, y_val, _ = ds.get_features_labels("val")
    X_test, y_test, features = ds.get_features_labels("test")

    if X_train is None:
        print("❌ Error: Ejecuta primero la opción 1 del menú principal para generar los datos.")
        return

    # 2. Eliminar la variable "trampa"
    feature_to_drop = 'is_near_fire'
    if feature_to_drop in X_train.columns:
        print(f"✂️  Eliminando variable: {feature_to_drop}")
        X_train = X_train.drop(columns=[feature_to_drop])
        X_val = X_val.drop(columns=[feature_to_drop])
        X_test = X_test.drop(columns=[feature_to_drop])
    else:
        print(f"⚠️ Variable {feature_to_drop} no encontrada. ¿Ya se eliminó?")

    # 3. Entrenar XGBoost
    print("🚀 Entrenando modelo (esto será rápido)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Evaluar
    print("\n📊 Resultados en TEST (Sin is_near_fire):")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= 0.5).astype(int)
    
    metrics = calculate_fire_metrics(y_test, test_pred, test_probs)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Guardar importancia de variables nueva
    imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 Top 5 Variables más importantes AHORA:")
    print(imp.head(5))

    joblib.dump(model, "models/SpainXGBoost_Ablated.pkl")
    print("\n💾 Modelo guardado como 'models/SpainXGBoost_Ablated.pkl'")

if __name__ == "__main__":
    main()
