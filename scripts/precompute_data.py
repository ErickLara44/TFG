import sys
from pathlib import Path
import xarray as xr
import torch
import argparse
import numpy as np
import pandas as pd
import gc
import shutil

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import config
from src.data.data_prop_improved import create_train_val_test_split, create_year_split, create_spatial_split
from src.data.data_ignition_improved import create_precomputed_ignition_datasets, create_ignition_datasets # Added create_ignition_datasets
from src.data.preprocessing import compute_derived_features

def main():
    parser = argparse.ArgumentParser(description="Pre-computar datasets para entrenamiento (Chunked)")
    parser.add_argument("--split", type=str, default="temporal_strict", 
                        choices=["random", "temporal_strict", "spatial"], 
                        help="Estrategia de split: random, temporal_strict (2018-2021 vs 2022), spatial (West vs East)")
    parser.add_argument("--region", type=str, default="east", help="Región de validación para split espacial (default: east)")
    parser.add_argument("--samples_per_year", type=int, default=500, help="Muestras a generar por año/chunk para entrenamiento")
    parser.add_argument("--all_fires", action="store_true", help="Si se activa, guarda TODOS los incendios disponibles (sin límite)")
    parser.add_argument("--start_year", type=int, default=2009, help="Año de inicio para procesar (default: 2009)")
    parser.add_argument("--end_year", type=int, default=2022, help="Año de fin para procesar (default: 2022)")
    # Ratios neg/pos por split. Default 2:1 (benchmark-style; SOTA usa 1:1 o ligero desbalance).
    parser.add_argument("--train_ratio", type=float, default=2.0, help="Ratio neg/pos en train (default: 2.0)")
    parser.add_argument("--val_ratio", type=float, default=2.0, help="Ratio neg/pos en val (default: 2.0)")
    parser.add_argument("--test_ratio", type=float, default=2.0, help="Ratio neg/pos en test benchmark (default: 2.0)")
    # Si se activa, genera un split adicional 'test_deployment' con distribución cercana a la real.
    parser.add_argument("--test_deployment_ratio", type=float, default=None,
                        help="Si se define, genera un split adicional 'test_deployment' con este ratio (ej. 100.0 para ~deployment real)")
    args = parser.parse_args()

    print(f"🚀 Iniciando pre-computación de datos (CHUNKED MODE)...")
    print(f"   Estrategia de Split: {args.split.upper()}")
    print(f"   Modo 'A TOPE' (All Fires): {args.all_fires}")
    print(f"📂 Datacube Source: {config.DATACUBE_PATH}")
    
    output_dir_base = config.DATA_DIR / "processed" / f"patches_{args.split}"
    if args.split == "spatial":
        output_dir_base = config.DATA_DIR / "processed" / f"patches_{args.split}_{args.region}"
    
    if args.all_fires:
        output_dir_base = config.DATA_DIR / "processed" / f"patches_{args.split}_FULL"
    
    # Limpiar directorio previo para evitar mezclas
    if output_dir_base.exists():
        print(f"🧹 Limpiando directorio de salida: {output_dir_base}")
        shutil.rmtree(output_dir_base)
    output_dir_base.mkdir(parents=True, exist_ok=True)

    # 1. Análisis preliminar de años (Lazy Load)
    try:
        ds_meta = xr.open_dataset(config.DATACUBE_PATH)
        all_years = sorted(list(set(pd.to_datetime(ds_meta.time.values).year)))
        print(f"📅 Años disponibles en Datacube: {all_years}")
        ds_meta.close()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {config.DATACUBE_PATH}")
        sys.exit(1)

    # 2. Definir Configuración de Split Global
    # ---------------------------------------
    # Mapeo de años a qué split pertenecen
    years_map = {}
    
    # Rango de años definido por usuario
    target_years_range = range(args.start_year, args.end_year + 1)
    
    if args.split == "temporal_strict":
        # Estrategia Temporal Estricta Extendida
        # Train: 2008-2020
        # Val: 2021-2022
        # Test: 2023-2024
        
        for y in target_years_range:
            if y <= 2020:
                years_map[y] = 'train'
            elif y <= 2022:
                years_map[y] = 'val'
            else:
                # 2023, 2024 -> Test
                years_map[y] = 'test'
        
    elif args.split == "spatial":
        # Todos los años contribuye a todos los splits (la mascara espacial decide)
        for y in target_years_range:
            years_map[y] = ['train', 'val', 'test'] 
            
    else: # Random
        for y in target_years_range:
            years_map[y] = ['train', 'val', 'test']

    # Contadores globales de índices para no sobreescribir archivos
    global_indices = {'train': 0, 'val': 0, 'test': 0}
    # Ratios por split y deployment split opcional
    balance_ratios = {'train': args.train_ratio, 'val': args.val_ratio, 'test': args.test_ratio}
    include_deployment = args.test_deployment_ratio is not None
    if include_deployment:
        balance_ratios['test_deployment'] = args.test_deployment_ratio
        global_indices['test_deployment'] = 0
        print(f"   🧪 test_deployment split activado (ratio neg/pos={args.test_deployment_ratio})")
    
    # 3. Procesamiento por Años (Chunking)
    # ------------------------------------
    years_to_process = sorted(list(years_map.keys()))
    print(f"✅ Años a procesar: {years_to_process}")
    
    import calendar

    for year in years_to_process:
        print(f"\n⚡ Procesando Año: {year}...")
        
        # Iterar por meses para ahorrar RAM
        for month in range(1, 13):
            # Definir rango del mes actual
            last_day = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{last_day}"
            
            # Margen: 15 días antes del inicio del mes
            # (Calculado con pandas para facilitar resta de fechas)
            start_dt = pd.to_datetime(start_date)
            buffer_dt = start_dt - pd.Timedelta(days=15)
            buffer_date = str(buffer_dt).split(" ")[0]
            
            print(f"   📅 Procesando Mes: {year}-{month:02d} (Buffer desde {buffer_date})")

            try:
                # Lazy load y slice
                ds_full = xr.open_dataset(config.DATACUBE_PATH)
                
                # Verificar si buffer existe
                available_start = pd.to_datetime(ds_full.time.values[0])
                available_end = pd.to_datetime(ds_full.time.values[-1])
                
                slice_start = buffer_date
                if pd.to_datetime(buffer_date) < available_start:
                    slice_start = str(available_start).split(" ")[0]
                
                # Si el mes está fuera del rango del dataset, saltar
                if pd.to_datetime(start_date) > available_end:
                     ds_full.close()
                     continue

                # Cargar chunk MENSUAL (+ buffer)
                # -------------------------------------
                print(f"   ⏳ Cargando chunk {slice_start} a {end_date} en memoria RAM...")
                ds_chunk = ds_full.sel(time=slice(slice_start, end_date)).load()
                ds_full.close()
                
                if len(ds_chunk.time) == 0:
                     print("   ⚠️ Chunk vacío, saltando.")
                     continue
                
                # 3.2 Feature Engineering (En RAM pequeña)
                print("   🛠️ Calculando features derivadas...")
                ds_chunk = compute_derived_features(ds_chunk)
                
                # 3.3 Generar Datasets para este chunk
                splits_for_year = years_map[year]
                if isinstance(splits_for_year, str): splits_for_year = [splits_for_year]
                
                # Indices válidos (SOLO los del mes actual, sin buffer)
                valid_times = ds_chunk.time.sel(time=slice(start_date, end_date)).values
                valid_indices = [i for i, t in enumerate(ds_chunk.time.values) if t in valid_times]
                
                if len(valid_indices) == 0:
                    print("   ⚠️ Sin índices válidos en este mes.")
                    del ds_chunk
                    gc.collect()
                    continue

                current_splits = {'train': [], 'val': [], 'test': []}
                if include_deployment:
                    current_splits['test_deployment'] = []
                for split_name in splits_for_year:
                    # Si el split no está en nuestra lista esperada (ej "train_val"), mapearlo o ignorarlo?
                    # Asumimos que years_map devuelve 'train', 'val', 'test'
                    if split_name in current_splits:
                        current_splits[split_name] = [{'time_index': idx, 'split': split_name} for idx in valid_indices]
                    # Espejo: los años de 'test' alimentan también 'test_deployment'
                    if include_deployment and split_name == 'test':
                        current_splits['test_deployment'] = [{'time_index': idx, 'split': 'test_deployment'} for idx in valid_indices]
                
                # 3.4 Generar (O GUARDAR) Datasets
                # -----------------------------------------------
                # Distribuir muestras proporcionalmente (aprox 500/12 por mes si es random, o todo si es a tope)
                # Si es ALL FIRES, el limite no importa.
                # Si es fixed samples, dividimos entre 12.
                samples_train_limit = max(1, args.samples_per_year // 12)
                samples_val_limit = max(1, (args.samples_per_year // 4) // 12)
                
                # SI ES ALL FIRES, PASAMOS None
                max_fires = None if args.all_fires else 10

                print(f"   🔥 Generando parches Mes {month} (Max Fires: {max_fires})...")
                
                datasets_dict = create_ignition_datasets(
                    ds_chunk, current_splits,
                    samples_train=samples_train_limit,
                    samples_val=samples_val_limit,
                    samples_test=samples_val_limit,
                    balance_ratios=balance_ratios,
                    spatial_masks=None,
                    max_fires_per_day=max_fires
                )
                
                # Recalcular spatial masks
                if args.split == "spatial":
                     s_split = create_spatial_split(ds_chunk, val_region=args.region, split_ratio=0.2)
                     for key in ('train', 'val', 'test'):
                         if key in datasets_dict:
                             datasets_dict[key].spatial_mask = s_split[key]
                     if 'test_deployment' in datasets_dict:
                         datasets_dict['test_deployment'].spatial_mask = s_split['test']

                # Loop de guardado
                for split_name, dataset in datasets_dict.items():
                    split_dir = output_dir_base / split_name
                    split_dir.mkdir(parents=True, exist_ok=True)
                    
                    start_idx = global_indices[split_name]
                    from src.data.data_ignition_improved import precompute_patches
                    
                    saved_count = precompute_patches(dataset, split_dir, start_index=start_idx)
                    global_indices[split_name] += saved_count
                
                # Limpiar memoria del MES
                del ds_chunk
                del datasets_dict
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Error procesando mes {year}-{month}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"   ✅ Año {year} completado.")

    print("\n✨ ¡Pre-computación CHUNKED finalizada con éxito!")
    print(f"   Total Train estimadas:           {global_indices['train']}")
    print(f"   Total Val estimadas:             {global_indices['val']}")
    print(f"   Total Test estimadas:            {global_indices['test']}")
    if include_deployment:
        print(f"   Total Test_deployment estimadas: {global_indices['test_deployment']}")
    
    # Guardar config
    with open(output_dir_base / "split_config.txt", "w") as f:
        f.write(f"Split Strategy: {args.split}\n")
        f.write(f"Method: Chunked (Year-by-Year)\n")
        f.write(f"Region: {args.region}\n")

if __name__ == "__main__":
    main()
