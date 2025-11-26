from sklearn.model_selection import TimeSeriesSplit

def temporal_validation(model_class, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []

    for train_idx, val_idx in tscv.split(X):
        train_X, val_X = X.iloc[train_idx], X.iloc[val_idx]
        train_y, val_y = y.iloc[train_idx], y.iloc[val_idx]
        model = model_class()
        val_pred, val_pred_proba = model.train_with_validation_strategy(train_X, train_y, val_X, val_y)
        metrics = model._calculate_fire_metrics(val_y, val_pred, val_pred_proba)
        metrics_list.append(metrics)
    return metrics_list