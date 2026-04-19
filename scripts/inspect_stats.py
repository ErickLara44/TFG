
import pickle
import sys
import os

stats_path = "data/processed/iberfire_normalization_stats.pkl"

try:
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    
    print("✅ Stats loaded successfully!")
    print(f"Keys: {stats.keys()}")
    
    vars_list = stats.get('vars', [])
    mean_arr = stats.get('mean')
    std_arr = stats.get('std')
    
    print("\nSTATS MAPPING:")
    if isinstance(vars_list, list) or isinstance(vars_list, np.ndarray):
        for i, var_name in enumerate(vars_list):
            m = mean_arr[i] if i < len(mean_arr) else None
            s = std_arr[i] if i < len(std_arr) else None
            print(f"  {var_name}: Mean={m}, Std={s}")
    else:
        print("Could not parse vars list.")
            
except Exception as e:
    print(f"❌ Error loading stats: {e}")
