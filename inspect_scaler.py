import pickle

try:
    with open('data/processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler type:", type(scaler))
    print("Features expected:", getattr(scaler, 'n_features_in_', 'Unknown'))
    if hasattr(scaler, 'mean_'):
        print("Mean:", scaler.mean_)
    if hasattr(scaler, 'scale_'):
        print("Scale:", scaler.scale_)
    if hasattr(scaler, 'feature_names_in_'):
        print("Feature Names:", scaler.feature_names_in_)
except Exception as e:
    print(e)
