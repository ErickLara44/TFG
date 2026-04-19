import pickle

with open('data/processed/iberfire_normalization_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

print("Keys:", stats.keys())
for k in stats.keys():
    if k not in ['mean', 'std']:
        print(f"{k}:", stats[k])
