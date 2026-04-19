import asyncio
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.api.data_fetcher import get_tensor_for_point
from src.api.inference import predict_ignition

async def debug_point():
    lat = 40.1740
    lon = -5.9953
    date = "2025-08-18"
    
    print(f"Requesting data for Point: {lat}, {lon} at {date}")
    tensor_np, features = await get_tensor_for_point(lat, lon, date)
    
    if "error" in features:
        print("Error fetching features:", features["error"])
        return

    print("--- Extracted Features ---")
    for k, v in features["features"].items():
        print(f"  {k}: {v}")

    print("\n--- Tensor Stats ---")
    print("Shape:", tensor_np.shape)
    print("Mean:", tensor_np.mean())
    print("Std:", tensor_np.std())
    print("NaNs:", np.isnan(tensor_np).sum())

    res = await predict_ignition(tensor_np, features)
    print("\n--- Prediction ---")
    print(res)

if __name__ == "__main__":
    asyncio.run(debug_point())
