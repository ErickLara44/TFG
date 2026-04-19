import pickle
import numpy as np
import sys

try:
    with open('data/processed/iberfire_normalization_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    print("Vars:", stats.get('variables', []))
    print("Mean:", stats['mean'])
    print("Std:", stats['std'])
except Exception as e:
    print(e)
