
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import joblib
import os
import sys

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.data_tab import SpainCubeFireDataset

def load_data():
    print("📦 Cargando datos de test normalizados...")
    # Usamos la clase dataset para facilitar la carga, aunque podríamos leer parquet directo
    # Instanciamos dummy solo para usar el método
    ds = SpainCubeFireDataset("dummy.nc", output_dir="data/processed/tabular")
    X_test, y_test, features = ds.get_features_labels("test")
    return X_test, y_test, features

def load_models():
    models = {}
    model_files = {
        "XGBoost": "models/SpainXGBoost.pkl",
        "RandomForest": "models/SpainRandomForest.pkl",
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"📂 Cargando {name}...")
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                print(f"❌ Error cargando {name}: {e}")
        else:
            print(f"⚠️ Modelo {name} no encontrado en {path}")
            
    return models

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        try:
            # Obtener probabilidades
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)[:, 1]
            else:
                continue
                
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        except Exception as e:
            print(f"⚠️ Error ploteando ROC para {name}: {e}")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Specificity)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensitivity)')
    plt.title('Curvas ROC - Comparativa de Modelos')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    output_path = "roc_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"💾 Guardado: {output_path}")
    plt.close()

def plot_confusion_matrices(models, X_test, y_test):
    n_models = len(models)
    if n_models == 0: return

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    for idx, (name, model) in enumerate(models.items()):
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Normalizar para ver porcentajes si se quiere, aquí usamos raw counts pero con anotaciones claras
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
            axes[idx].set_title(f'Matriz de Confusión - {name}')
            axes[idx].set_xlabel('Predicción')
            axes[idx].set_ylabel('Real')
            axes[idx].set_xticklabels(['No Fuego', 'Fuego'])
            axes[idx].set_yticklabels(['No Fuego', 'Fuego'])
        except Exception as e:
            print(f"⚠️ Error ploteando CM para {name}: {e}")
            
    output_path = "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"💾 Guardado: {output_path}")
    plt.close()

def plot_feature_importance_comparison(models, features):
    # Solo para modelos que tengan feature_importances_
    # CatBoost y XGBoost lo tienen, RF también
    
    dfs = []
    
    for name, model in models.items():
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                # Si es CatBoost, a veces es diferente el acceso, pero sklearn API suele cumplir
                df = pd.DataFrame({'feature': features, 'importance': imp, 'model': name})
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ No se pudo extraer importancia de {name}: {e}")
            
    if not dfs: return

    df_all = pd.concat(dfs)
    
    # Calcular top features promedio para ordenar
    top_features = df_all.groupby('feature')['importance'].mean().sort_values(ascending=False).head(15).index
    
    df_plot = df_all[df_all['feature'].isin(top_features)]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', hue='model', data=df_plot, order=top_features)
    plt.title('Top 15 Variables más Importantes por Modelo')
    plt.tight_layout()
    
    output_path = "feature_importance_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"💾 Guardado: {output_path}")
    plt.close()

def main():
    try:
        X_test, y_test, features = load_data()
        if X_test is None:
            print("❌ No se pudieron cargar los datos de test.")
            return

        models = load_models()
        if not models:
            print("❌ No se cargaron modelos.")
            return

        print("\n📊 Generando Curvas ROC...")
        plot_roc_curves(models, X_test, y_test)
        
        print("\n📊 Generando Matrices de Confusión...")
        plot_confusion_matrices(models, X_test, y_test)
        
        print("\n📊 Generando Importancia de Variables...")
        plot_feature_importance_comparison(models, features)
        
        print("\n✅ Visualización completada.")
        
    except Exception as e:
        print(f"❌ Error global: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
