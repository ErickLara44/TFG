# tabular_models.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib, os

# ------------------------------------------------------------
# MÉTRICAS COMUNES
# ------------------------------------------------------------
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

def print_metrics(name, metrics):
    print(f"📊 Métricas de validación {name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# ------------------------------------------------------------
# PLOT FEATURE IMPORTANCE
# ------------------------------------------------------------
def plot_feature_importance(df_importance, top_n=20, title="Importancia de características"):
    imp = df_importance.head(top_n)
    plt.figure(figsize=(8,6))
    plt.barh(imp['feature'], imp['importance'])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importancia")
    plt.show()

# ------------------------------------------------------------
# PLOT MAPA DE PREDICCIÓN
# ------------------------------------------------------------
def plot_fire_map(pred_probs, coords, threshold=0.5):
    """
    pred_probs: array de probabilidades
    coords: DataFrame con columnas ['x','y']
    """
    plt.figure(figsize=(8,6))
    sc = plt.scatter(coords['x'], coords['y'], c=pred_probs, cmap="hot", s=20)
    plt.colorbar(sc, label="Probabilidad de incendio")
    plt.title("Mapa de predicción de incendios")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.show()

# ------------------------------------------------------------
# BASE CLASS para guardar/cargar modelos
# ------------------------------------------------------------
class BaseFirePredictor:
    def _save_model(self, name):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, f"models/{name}.pkl")
        if self.feature_importance is not None:
            joblib.dump(self.feature_importance, f"models/{name}_feat_importance.pkl")
        print(f"💾 Modelo {name} guardado en /models")

    def load_model(self, name):
        self.model = joblib.load(f"models/{name}.pkl")
        try:
            self.feature_importance = joblib.load(f"models/{name}_feat_importance.pkl")
        except FileNotFoundError:
            self.feature_importance = None
        print(f"📂 Modelo {name} cargado desde /models")

# ------------------------------------------------------------
# MODELOS
# ------------------------------------------------------------
class SpainRandomForestPredictor(BaseFirePredictor):
    def __init__(self, n_estimators=500, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state
        )
        self.feature_importance = None
    
    def train_with_validation_strategy(self, train_X, train_y, val_X, val_y):
        self.model.fit(train_X, train_y)
        val_pred = self.model.predict(val_X)
        val_pred_proba = self.model.predict_proba(val_X)[:,1]
        metrics = calculate_fire_metrics(val_y, val_pred, val_pred_proba)
        self.feature_importance = pd.DataFrame({
            'feature': train_X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print_metrics("Random Forest", metrics)
        self._save_model("SpainRandomForest")
        return metrics
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

class SpainXGBoostPredictor(BaseFirePredictor):
    def __init__(self, random_state=42):
        self.model = None
        self.feature_importance = None
        self.random_state = random_state

    def train_with_validation_strategy(self, train_X, train_y, val_X, val_y, use_gpu=False):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'n_jobs': -1,
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': len(train_y[train_y == 0]) / len(train_y[train_y == 1])
        }
        if use_gpu:
            params['tree_method'] = 'gpu_hist'

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)

        val_pred = self.model.predict(val_X)
        val_pred_proba = self.model.predict_proba(val_X)[:,1]
        metrics = calculate_fire_metrics(val_y, val_pred, val_pred_proba)
        self.feature_importance = pd.DataFrame({
            'feature': train_X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print_metrics("XGBoost", metrics)
        self._save_model("SpainXGBoost")
        return metrics
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

class SpainLightGBMPredictor(BaseFirePredictor):
    def __init__(self, random_state=42):
        self.model = None
        self.feature_importance = None
        self.random_state = random_state

    def train_with_validation_strategy(self, train_X, train_y, val_X, val_y):
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1
        )
        self.model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)
        val_pred = self.model.predict(val_X)
        val_pred_proba = self.model.predict_proba(val_X)[:,1]
        metrics = calculate_fire_metrics(val_y, val_pred, val_pred_proba)
        self.feature_importance = pd.DataFrame({
            'feature': train_X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print_metrics("LightGBM", metrics)
        self._save_model("SpainLightGBM")
        return metrics
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

class SpainCatBoostPredictor(BaseFirePredictor):
    def __init__(self, random_state=42):
        self.model = None
        self.feature_importance = None
        self.random_state = random_state

    def train_with_validation_strategy(self, train_X, train_y, val_X, val_y):
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=self.random_state,
            verbose=False,
            class_weights=[1.0, len(train_y[train_y == 0]) / len(train_y[train_y == 1])]
        )
        self.model.fit(train_X, train_y, eval_set=(val_X, val_y))
        val_pred = self.model.predict(val_X)
        val_pred_proba = self.model.predict_proba(val_X)[:,1]
        metrics = calculate_fire_metrics(val_y, val_pred, val_pred_proba)
        self.feature_importance = pd.DataFrame({
            'feature': train_X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print_metrics("CatBoost", metrics)
        self._save_model("SpainCatBoost")
        return metrics
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]