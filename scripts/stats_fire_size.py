
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

def analyze_fire_sizes(data_dir):
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.pt")))
    
    print(f"📦 Analizando {len(files)} muestras en {data_dir}...")
    
    fire_pixels_list = []
    
    for f in tqdm(files):
        try:
            data = torch.load(f, weights_only=False)
            # y es la máscara de fuego futura (1, H, W)
            # x es la secuencia pasada. x[-1, -1] es la máscara de fuego actual.
            
            # Miremos el target (y) para ver cuánto ocupa el fuego a predecir
            y = data['y'] # (1, H, W)
            fire_pixels = y.sum().item()
            fire_pixels_list.append(fire_pixels)
            
        except Exception as e:
            continue
            
    fire_pixels_arr = np.array(fire_pixels_list)
    
    # 1 pixel = 1km x 1km = 100 hectáreas (aprox, usuario dijo resolucion 1km)
    # Si es 500m x 500m -> 0.25 km2 = 25 hectáreas
    
    mean_px = fire_pixels_arr.mean()
    max_px = fire_pixels_arr.max()
    min_px = fire_pixels_arr.min()
    
    print("\n🔥 ESTADÍSTICAS DE TAMAÑO DE FUEGO (Dataset):")
    print(f"   Muestras: {len(fire_pixels_arr)}")
    print(f"   Media (Píxeles): {mean_px:.2f}")
    print(f"   Máximo (Píxeles): {max_px:.0f}")
    print(f"   Mínimo (Píxeles): {min_px:.0f}")
    
    print("\n🌍 INTERPRETACIÓN (Asumiendo 1 pxl = 1 km²):")
    print(f"   Media: {mean_px:.2f} km² ({mean_px*100:.0f} ha)")
    print(f"   Máximo: {max_px:.0f} km² ({max_px*100:.0f} ha)")
    
    # Percentiles
    p50 = np.percentile(fire_pixels_arr, 50)
    p90 = np.percentile(fire_pixels_arr, 90)
    p99 = np.percentile(fire_pixels_arr, 99)
    
    print(f"\n📊 DISTRIBUCIÓN:")
    print(f"   50% de incendios son menores a: {p50:.0f} px ({p50:.0f} km²)")
    print(f"   90% de incendios son menores a: {p90:.0f} px ({p90:.0f} km²)")
    print(f"   99% de incendios son menores a: {p99:.0f} px ({p99:.0f} km²)")

def main():
    train_dir = "data/processed/patches/spread_224/train"
    analyze_fire_sizes(train_dir)

if __name__ == "__main__":
    main()
