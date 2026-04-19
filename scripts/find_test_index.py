
import torch
import sys
import os
from pathlib import Path

def find_test_index_cache():
    print("🕵️‍♂️ Buscando caché de índices para el TEST set (69 muestras)...")
    
    cache_dir = "data/processed/cache_spread"
    files = list(Path(cache_dir).glob("*.pt"))
    
    for f in files:
        try:
            indices = torch.load(f)
            print(f"📄 {f.name}: {len(indices)} muestras")
            
            if len(indices) == 69:
                print(f"   ✅ ¡ENCONTRADO! {f.name} parece ser el de TEST.")
                
                # Extraer info para muesra 21
                sample_info = indices[21]
                print(f"   📍 Muestra 21: {sample_info}")
                
        except Exception as e:
            print(f"   ❌ Error leyendo {f.name}: {e}")

if __name__ == "__main__":
    find_test_index_cache()
