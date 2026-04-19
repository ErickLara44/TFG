
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data.data_prop_improved import generate_temporal_splits

def identify_sample_21():
    print("🕵️‍♂️ Identificando origen de Muestra 21 (Replicando lógica de split)...")
    
    # 1. Cargar Datacube (Lazy)
    datacube_path = config.DATACUBE_PATH
    if not os.path.exists(datacube_path):
        print(f"❌ No se encuentra: {datacube_path}")
        return

    ds = xr.open_dataset(datacube_path)
    
    # 2. Generar Splits (Igual que prepare_patches)
    print("✂️ Generando splits temporal...")
    splits = generate_temporal_splits(ds, strict=True)
    
    # 3. Obtener indices de TEST
    test_indices = splits.get('test', [])
    print(f"   Indices Test Crudos: {len(test_indices)}")
    
    # 4. Filtrar por Fuego (Igual que prepare_patches)
    # Esto es lo lento, porque hay que mirar burned_area en cada indice
    # PERO, podemos usar la caché si existe.
    # prepare_patches usa SpreadDataset con filter_fire_samples=True.
    # Vamos a instanciar SpreadDataset para re-usar su lógica de filtrado exacto.
    
    from src.data.data_prop_improved import SpreadDataset
    
    # OJO: feature_vars da igual para el índice
    dataset = SpreadDataset(
        ds, 
        test_indices, 
        temporal_context=3, 
        filter_fire_samples=True, 
        preload_ram=False
    )
    
    print(f"   Indices Test Filtrados (con Fuego): {len(dataset)}")
    
    # 5. Ordenar por Tiempo (Igual que save_patches)
    # dataset.indices es una lista de dicts
    dataset.indices.sort(key=lambda x: x['time_index'])
    
    # 6. Seleccionar el 21
    if len(dataset) > 21:
        target = dataset.indices[21]
        print("\n✅ KEY INSPECTION:")
        print(target.keys())
        
        # Adapt keys
        t_key = 'time_index' if 'time_index' in target else 'time'
        y_key = 'y_idx' if 'y_idx' in target else 'y'
        x_key = 'x_idx' if 'x_idx' in target else 'x'
        
        t_idx = target[t_key]
        y_idx = target.get(y_key, target.get('y_center'))
        x_idx = target.get(x_key, target.get('x_center'))
        
        print("\n✅ MUESTRA 21 IDENTIFICADA:")
        print(f"   - Time Index: {t_idx}")
        print(f"   - Timestamp: {ds.time.values[t_idx]}")
        
        # Si no hay coordenadas en el índice (caso temporal split), hay que BUSCARLAS.
        # El SpreadDataset original buscaba fuegos en __init__ si no se daban indices.
        # Aquí sabemos el tiempo (5173), así que busquemos dónde hay fuego en ese tiempo.
        
        if y_idx is None or x_idx is None:
            print("⚠️ Coordenadas no en índice. Buscando fuego en ese timestep...")
            ba_slice = ds.is_fire.isel(time=t_idx).values
            # Buscar componentes conectados o simplemente píxeles de fuego
            fire_pixels = np.argwhere(ba_slice > 0.5)
            
            if len(fire_pixels) > 0:
                # Tomamos el centro del primer cluster grande o simplemente el centro de masa
                y_center = int(fire_pixels[:, 0].mean())
                x_center = int(fire_pixels[:, 1].mean())
                print(f"✅ Fuego encontrado en: y={y_center}, x={x_center}")
                y_idx = y_center
                x_idx = x_center
            else:
                print("❌ No se encontró fuego en ese timestep (¿Quizás era el día siguiente?)")
                return

        # 7. Validar contenido
        # Extraer ventana real (t y t+1)
        
        # Crop size 224
        half = 224 // 2
        y_slice = slice(y_idx - half, y_idx + half)
        x_slice = slice(x_idx - half, x_idx + half)
        
        # Cargar burned area t y t+1
        # t (Input)
        try:
            ba_t = ds.is_fire.isel(time=t_idx, y=y_slice, x=x_slice).values
            # t+1 (Target)
            ba_t1 = ds.is_fire.isel(time=t_idx+1, y=y_slice, x=x_slice).values
            
            print("\n📊 Validación Directa del Cubo:")
            print(f"   - Fuego t: {(ba_t > 0.5).sum()} px")
            print(f"   - Fuego t+1: {(ba_t1 > 0.5).sum()} px")
            overlap = ((ba_t > 0.5) & (ba_t1 > 0.5)).sum()
            print(f"   - Superposición: {overlap} px")
            
            if overlap == 0:
                print("   ⚠️ CONFIRMADO: Salto sin superposición en el cubo original.")
                
            # Guardar visualización RAW
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(ba_t, cmap='Reds')
            plt.title(f"Cubo Original t (idx={t_idx})\nVariable: is_fire")
            plt.subplot(1, 2, 2)
            plt.imshow(ba_t1, cmap='Oranges')
            plt.title(f"Cubo Original t+1 (idx={t_idx+1})\nVariable: is_fire")
            
            plt.savefig("outputs/debug/cube_verification_21.png")
            print("📸 Guardado: outputs/debug/cube_verification_21.png")
            
        except Exception as e:
            print(f"❌ Error extrayendo slice: {e}")

if __name__ == "__main__":
    identify_sample_21()
