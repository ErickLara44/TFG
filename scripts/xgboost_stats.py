"""
XGBoost & Random Forest - Metricas y Visualizaciones para TFG
Genera: ROC, PR curve, confusion matrix, feature importance,
        threshold analysis, calibration, y tabla resumen.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    roc_auc_score, f1_score
)
from sklearn.calibration import calibration_curve
from src.data.data_tab import SpainCubeFireDataset
from src.models.XGBoost import calculate_fire_metrics

# ── Config ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/tabular_stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Estilo global
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


# ── Carga de datos ──────────────────────────────────────────────
def load_data():
    ds = SpainCubeFireDataset("dummy.nc", output_dir="data/processed/tabular")
    splits = {}
    for split in ["train", "val", "test"]:
        X, y, feats = ds.get_features_labels(split)
        if 'is_near_fire' in X.columns:
            X = X.drop(columns=['is_near_fire'])
        splits[split] = (X, y)
    return splits, [f for f in feats if f != 'is_near_fire']


def load_models():
    models = {}
    for name, path in [("XGBoost", "models/SpainXGBoost.pkl"),
                       ("Random Forest", "models/SpainRandomForest.pkl")]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


# ── 1. Curvas ROC ───────────────────────────────────────────────
def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='Curvas ROC — Test Set', xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png")
    plt.close()
    print(f"  Guardado: roc_curves.png")


# ── 2. Curvas Precision-Recall ──────────────────────────────────
def plot_pr(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2, label=f'{name} (AP = {ap:.4f})')

    prevalence = y_test.mean()
    ax.axhline(prevalence, color='k', ls='--', lw=1, alpha=0.5,
               label=f'Baseline (prevalencia = {prevalence:.2f})')
    ax.set(xlabel='Recall (Sensitivity)', ylabel='Precision',
           title='Curvas Precision-Recall — Test Set', xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pr_curves.png")
    plt.close()
    print(f"  Guardado: pr_curves.png")


# ── 3. Matrices de Confusion (threshold optimo) ────────────────
def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Encuentra el threshold que maximiza F1."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_score = 0.5, 0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


def plot_confusion_matrices(models, X_test, y_test):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        opt_t, opt_f1 = find_optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= opt_t).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        # Normalizado por fila (porcentajes)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        labels = np.array([[f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)"
                           for j in range(2)] for i in range(2)])

        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=axes[idx],
                    cbar=False, xticklabels=['No Fuego', 'Fuego'],
                    yticklabels=['No Fuego', 'Fuego'])
        axes[idx].set_title(f'{name}\n(threshold={opt_t:.2f}, F1={opt_f1:.4f})')
        axes[idx].set_xlabel('Prediccion')
        axes[idx].set_ylabel('Real')

    fig.suptitle('Matrices de Confusion — Threshold Optimo (max F1)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", bbox_inches='tight')
    plt.close()
    print(f"  Guardado: confusion_matrices.png")


# ── 4. Feature Importance (top 20) ─────────────────────────────
def plot_feature_importance(models, features):
    dfs = []
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            dfs.append(pd.DataFrame({
                'feature': features[:len(imp)],
                'importance': imp,
                'model': name
            }))

    if not dfs:
        return

    df = pd.concat(dfs)
    top = df.groupby('feature')['importance'].mean().sort_values(ascending=False).head(20).index
    df_top = df[df['feature'].isin(top)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', hue='model', data=df_top, order=top, ax=ax)
    ax.set_title('Top 20 Features por Importancia')
    ax.set_xlabel('Importancia')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()
    print(f"  Guardado: feature_importance.png")


# ── 5. Threshold Analysis ──────────────────────────────────────
def plot_threshold_analysis(models, X_test, y_test):
    thresholds = np.arange(0.05, 0.95, 0.01)
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics_at_t = {'Sensitivity': [], 'Specificity': [],
                        'Precision': [], 'F1': []}

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
            metrics_at_t['Sensitivity'].append(sens)
            metrics_at_t['Specificity'].append(spec)
            metrics_at_t['Precision'].append(prec)
            metrics_at_t['F1'].append(f1)

        ax = axes[idx]
        for metric_name, values in metrics_at_t.items():
            ax.plot(thresholds, values, lw=2, label=metric_name)

        # Marcar threshold optimo F1
        best_idx = np.argmax(metrics_at_t['F1'])
        best_t = thresholds[best_idx]
        ax.axvline(best_t, color='red', ls='--', alpha=0.7,
                   label=f'Optimo F1={metrics_at_t["F1"][best_idx]:.3f} (t={best_t:.2f})')

        ax.set(xlabel='Threshold', ylabel='Score', title=f'{name} — Metricas vs Threshold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_analysis.png")
    plt.close()
    print(f"  Guardado: threshold_analysis.png")


# ── 6. Calibration Curve ───────────────────────────────────────
def plot_calibration(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=15,
                                                 strategy='quantile')
        ax.plot(mean_pred, frac_pos, 's-', lw=2, label=name)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfectamente calibrado')
    ax.set(xlabel='Probabilidad media predicha', ylabel='Fraccion de positivos',
           title='Curva de Calibracion — Test Set')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration.png")
    plt.close()
    print(f"  Guardado: calibration.png")


# ── 7. Distribucion de probabilidades ──────────────────────────
def plot_prob_distribution(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        ax = axes[idx]
        ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, label='No Fuego', color='steelblue', density=True)
        ax.hist(y_prob[y_test == 1], bins=50, alpha=0.6, label='Fuego', color='tomato', density=True)
        ax.set(xlabel='Probabilidad predicha', ylabel='Densidad',
               title=f'{name} — Distribucion de Probabilidades')
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prob_distribution.png")
    plt.close()
    print(f"  Guardado: prob_distribution.png")


# ── 8. Tabla resumen de metricas ────────────────────────────────
def generate_metrics_table(models, X_val, y_val, X_test, y_test):
    rows = []

    for name, model in models.items():
        for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_prob = model.predict_proba(X)[:, 1]

            # Threshold 0.5
            y_05 = (y_prob >= 0.5).astype(int)
            m05 = calculate_fire_metrics(y, y_05, y_prob)

            # Threshold optimo
            opt_t, _ = find_optimal_threshold(y, y_prob)
            y_opt = (y_prob >= opt_t).astype(int)
            m_opt = calculate_fire_metrics(y, y_opt, y_prob)

            rows.append({
                'Modelo': name, 'Split': split_name, 'Threshold': 0.5,
                **{k: round(v, 4) for k, v in m05.items()}
            })
            rows.append({
                'Modelo': name, 'Split': split_name, 'Threshold': opt_t,
                **{k: round(v, 4) for k, v in m_opt.items()}
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("RESUMEN DE METRICAS")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"\nGuardado: metrics_summary.csv")
    return df


# ── Main ────────────────────────────────────────────────────────
def main():
    print("Cargando datos...")
    splits, features = load_data()
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features: {len(features)}")

    print("\nCargando modelos...")
    models = load_models()
    if not models:
        print("No se encontraron modelos entrenados en models/")
        return

    print(f"Modelos: {list(models.keys())}")

    print("\nGenerando visualizaciones...")
    plot_roc(models, X_test, y_test)
    plot_pr(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models, features)
    plot_threshold_analysis(models, X_test, y_test)
    plot_calibration(models, X_test, y_test)
    plot_prob_distribution(models, X_test, y_test)
    df_metrics = generate_metrics_table(models, X_val, y_val, X_test, y_test)

    print(f"\nTodas las figuras guardadas en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
