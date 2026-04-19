"""
build_tabular_fast.py  –  Extracción RÁPIDA de dataset tabular para XGBoost / RF.

Estrategia:
  1. Cargar datacube con lazy loading (xarray)
  2. Crear máscara espacial de CCAA target UNA SOLA VEZ
  3. Escanear SOLO 'is_fire' para encontrar eventos de fuego (variable más ligera)
  4. Muestrear negativos (no-fuego) usando la máscara espacial
  5. Extraer features SOLO para esos puntos concretos (t, y, x)
  6. Guardar train/val/test como Parquet

Resultado: minutos en lugar de horas.
"""

import sys
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import gc
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS, VIRTUAL_TIME_VARS  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════
CCAA_NAMES = {
    1: "Andalucía", 3: "Asturias", 7: "Castilla y León",
    10: "C. Valenciana", 11: "Extremadura", 12: "Galicia",
}


def build_ccaa_mask(ds, ccaa_codes):
    """Crea máscara booleana 2D (y, x) de celdas en las CCAA objetivo."""
    ccaa = ds["AutonomousCommunities"]
    # Si tiene dimensión time, tomar el primer timestep (es estático)
    if "time" in ccaa.dims:
        ccaa = ccaa.isel(time=0)
    mask = np.isin(ccaa.values, ccaa_codes)
    print(f"   Celdas en CCAA target: {mask.sum():,} / {mask.size:,} ({mask.sum()/mask.size*100:.1f}%)")
    return mask


def scan_fires(ds, mask, start_year, end_year):
    """
    Escanea 'is_fire' y devuelve lista de (time_idx, y, x) donde hay fuego,
    filtrado por la máscara espacial.  Rápido porque solo lee UNA variable.
    """
    print("\n🔥 Escaneando eventos de fuego...")
    fire_var = ds["is_fire"]
    times = pd.to_datetime(ds.time.values)
    
    fire_events = []  # (time_idx, y_idx, x_idx, year)
    
    for t_idx in tqdm(range(len(times)), desc="Buscando fuegos"):
        year = times[t_idx].year
        if year < start_year or year > end_year:
            continue
        
        # Leer solo este timestep de is_fire (ligero)
        fire_slice = fire_var.isel(time=t_idx).values
        
        # Aplicar máscara espacial Y fuego
        fire_in_ccaa = (fire_slice > 0) & mask
        
        if fire_in_ccaa.any():
            ys, xs = np.where(fire_in_ccaa)
            for y, x in zip(ys, xs):
                fire_events.append((t_idx, int(y), int(x), year))
    
    print(f"   Total eventos de fuego en CCAA: {len(fire_events):,}")
    return fire_events


def sample_negatives(ds, mask, fire_events, neg_ratio, start_year, end_year):
    """
    Muestrea puntos (t, y, x) sin fuego, respetando la máscara espacial.
    """
    print(f"\n🌿 Muestreando negativos (ratio {neg_ratio}:1)...")
    
    n_neg = int(len(fire_events) * neg_ratio)
    
    times = pd.to_datetime(ds.time.values)
    fire_var = ds["is_fire"]
    
    # Coordenadas válidas (celdas en CCAA)
    valid_ys, valid_xs = np.where(mask)
    n_valid_pixels = len(valid_ys)
    
    # Índices temporales válidos
    valid_t_indices = [i for i, t in enumerate(times) 
                       if start_year <= t.year <= end_year]
    
    # Set de coordenadas con fuego para evitar colisiones
    fire_set = set((t, y, x) for t, y, x, _ in fire_events)
    
    neg_events = []
    rng = np.random.default_rng(42)
    
    attempts = 0
    max_attempts = n_neg * 10
    
    with tqdm(total=n_neg, desc="Sampling negatives") as pbar:
        while len(neg_events) < n_neg and attempts < max_attempts:
            # Batch sampling (mucho más rápido que uno a uno)
            batch_size = min(n_neg - len(neg_events), 5000)
            
            t_samples = rng.choice(valid_t_indices, size=batch_size)
            px_samples = rng.integers(0, n_valid_pixels, size=batch_size)
            
            for i in range(batch_size):
                t_idx = int(t_samples[i])
                y_idx = int(valid_ys[px_samples[i]])
                x_idx = int(valid_xs[px_samples[i]])
                year = times[t_idx].year
                
                if (t_idx, y_idx, x_idx) not in fire_set:
                    neg_events.append((t_idx, y_idx, x_idx, year))
                    pbar.update(1)
                    
                    if len(neg_events) >= n_neg:
                        break
                
                attempts += 1
    
    print(f"   Negativos muestreados: {len(neg_events):,}")
    return neg_events


def extract_features_for_points(ds, points, feature_vars):
    """
    Extrae features para una lista de puntos (t_idx, y_idx, x_idx, year).
    Devuelve un DataFrame listo para XGBoost.
    
    VERSIÓN MEMORY-SAFE: agrupa puntos por timestep y carga UN timestep
    a la vez por variable. Pico de RAM ~= tamaño de 1 slice espacial (~MB).
    """
    print(f"\n📊 Extrayendo {len(feature_vars)} features para {len(points):,} puntos...")
    
    times = pd.to_datetime(ds.time.values)
    
    # Separar variables del datacube vs virtuales
    cube_vars = [v for v in feature_vars if v not in VIRTUAL_TIME_VARS]
    virtual_vars = [v for v in feature_vars if v in VIRTUAL_TIME_VARS]
    
    # Pre-computar variables virtuales (calendario) para todos los timesteps
    doy = times.dayofyear.values.astype(np.float32)
    two_pi = 2.0 * np.pi
    virtual_cache = {
        'is_weekend': (times.dayofweek.values >= 5).astype(np.float32),
        'day_of_year_sin': np.sin(two_pi * doy / 365.25).astype(np.float32),
        'day_of_year_cos': np.cos(two_pi * doy / 365.25).astype(np.float32),
    }
    
    # Preparar arrays de índices
    t_indices = np.array([p[0] for p in points])
    y_indices = np.array([p[1] for p in points])
    x_indices = np.array([p[2] for p in points])
    years = np.array([p[3] for p in points])
    
    # Diccionario de resultados
    data = {
        'time': times[t_indices],
        'y_idx': y_indices,
        'x_idx': x_indices,
        'year': years,
    }
    
    # Extraer coordenadas reales x, y si existen
    if 'x' in ds.coords:
        x_coords = ds.x.values
        data['x'] = x_coords[x_indices]
    if 'y' in ds.coords:
        y_coords = ds.y.values
        data['y'] = y_coords[y_indices]
    
    # ──────────────────────────────────────────────────────────
    # AGRUPAR PUNTOS POR TIMESTEP (clave para bajo uso de RAM)
    # ──────────────────────────────────────────────────────────
    # Crear mapa: time_idx -> lista de posiciones en el array 'points'
    from collections import defaultdict
    time_to_positions = defaultdict(list)
    for pos, t in enumerate(t_indices):
        time_to_positions[int(t)].append(pos)
    
    unique_times = sorted(time_to_positions.keys())
    print(f"   Timesteps únicos a procesar: {len(unique_times)}")
    
    # Separar variables estáticas vs dinámicas
    static_vars = [v for v in cube_vars if v in ds and "time" not in ds[v].dims]
    dynamic_vars = [v for v in cube_vars if v in ds and "time" in ds[v].dims]
    missing_vars = [v for v in cube_vars if v not in ds.data_vars and v not in ds.coords]
    
    if missing_vars:
        print(f"   ⚠️ Variables no encontradas (skip): {missing_vars}")
    
    # ── ESTÁTICAS: cargar una vez (solo 1 slice 2D, ligero) ──
    for var in static_vars:
        vals = ds[var].values  # (Y, X) — pocos MB
        data[var] = vals[y_indices, x_indices].astype(np.float32)
    print(f"   ✅ {len(static_vars)} variables estáticas extraídas")
    
    # ── DINÁMICAS: iterar timestep a timestep ──
    # Pre-allocar arrays de resultado
    for var in dynamic_vars:
        data[var] = np.empty(len(points), dtype=np.float32)
    
    for t_idx in tqdm(unique_times, desc="Extrayendo por timestep"):
        positions = time_to_positions[t_idx]
        pos_arr = np.array(positions)
        ys = y_indices[pos_arr]
        xs = x_indices[pos_arr]
        
        # Cargar UN solo timestep de TODAS las variables dinámicas a la vez
        # Esto es un slice (Y, X) por variable — pocos MB en total
        ds_slice = ds[dynamic_vars].isel(time=t_idx)
        
        for var in dynamic_vars:
            slice_vals = ds_slice[var].values  # (Y, X)
            data[var][pos_arr] = slice_vals[ys, xs].astype(np.float32)
        
        del ds_slice
    
    gc.collect()
    
    # Extraer variables virtuales (calendario)
    for var in virtual_vars:
        if var in virtual_cache:
            data[var] = virtual_cache[var][t_indices]
    
    df = pd.DataFrame(data)
    
    # Limpiar NaN
    n_nan = df[cube_vars + list(virtual_vars)].isna().sum().sum()
    if n_nan > 0:
        print(f"   ⚠️ {n_nan} NaN encontrados, rellenando con 0")
        df = df.fillna(0)
    
    print(f"   ✅ DataFrame shape: {df.shape}")
    return df


def run_fast_pipeline(datacube_path, output_dir, start_year=2008, end_year=2024,
                      ccaa_target=None, neg_ratio=2.0,
                      train_year_max=2020, val_years=(2021, 2022),
                      test_years=(2023, 2024)):
    """Pipeline completo rápido."""
    
    if ccaa_target is None:
        ccaa_target = [12, 3, 7, 11, 1]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    val_years = set(val_years)
    test_years = set(test_years)
    
    feature_vars = list(DEFAULT_FEATURE_VARS)
    
    print("=" * 60)
    print("⚡ PIPELINE TABULAR RÁPIDO PARA XGBOOST")
    print("=" * 60)
    print(f"Datacube: {datacube_path}")
    print(f"Periodo: {start_year}-{end_year}")
    print(f"CCAA: {[CCAA_NAMES.get(c, c) for c in ccaa_target]}")
    print(f"Split: train ≤{train_year_max} | val {sorted(val_years)} | test {sorted(test_years)}")
    print(f"Ratio neg/pos: {neg_ratio}")
    print(f"Features: {len(feature_vars)}")
    print("=" * 60)
    
    # ──────────────────────────────────────────────
    # 1. ABRIR DATACUBE (lazy, no carga nada aún)
    # ──────────────────────────────────────────────
    print("\n📂 Abriendo datacube...")
    ds = xr.open_dataset(datacube_path)
    
    # ──────────────────────────────────────────────
    # 2. MÁSCARA ESPACIAL (una sola vez, rápido)
    # ──────────────────────────────────────────────
    print("\n🗺️  Construyendo máscara espacial...")
    mask = build_ccaa_mask(ds, ccaa_target)
    
    # ──────────────────────────────────────────────
    # 3. ESCANEAR FUEGOS (con caché en disco)
    # ──────────────────────────────────────────────
    cache_fire = output_path / "_cache_fire_events.pkl"
    cache_neg = output_path / "_cache_neg_events.pkl"
    
    if cache_fire.exists():
        print(f"\n📦 Cargando fuegos desde caché: {cache_fire}")
        with open(cache_fire, "rb") as f:
            fire_events = pickle.load(f)
        print(f"   {len(fire_events):,} eventos de fuego cargados")
    else:
        fire_events = scan_fires(ds, mask, start_year, end_year)
        if len(fire_events) == 0:
            print("❌ No se encontraron fuegos. Abortando.")
            ds.close()
            return
        # Guardar caché
        with open(cache_fire, "wb") as f:
            pickle.dump(fire_events, f)
        print(f"   💾 Caché guardada: {cache_fire}")
    
    # Stats por año
    fire_by_year = {}
    for _, _, _, yr in fire_events:
        fire_by_year[yr] = fire_by_year.get(yr, 0) + 1
    print("\n📅 Fuegos por año:")
    for yr in sorted(fire_by_year.keys()):
        print(f"   {yr}: {fire_by_year[yr]:,}")
    
    # ──────────────────────────────────────────────
    # 4. MUESTREAR NEGATIVOS (con caché en disco)
    # ──────────────────────────────────────────────
    if cache_neg.exists():
        print(f"\n📦 Cargando negativos desde caché: {cache_neg}")
        with open(cache_neg, "rb") as f:
            neg_events = pickle.load(f)
        print(f"   {len(neg_events):,} negativos cargados")
    else:
        neg_events = sample_negatives(ds, mask, fire_events, neg_ratio, start_year, end_year)
        with open(cache_neg, "wb") as f:
            pickle.dump(neg_events, f)
        print(f"   💾 Caché guardada: {cache_neg}")
    
    # ──────────────────────────────────────────────
    # 5. COMBINAR Y EXTRAER FEATURES
    # ──────────────────────────────────────────────
    all_points = fire_events + neg_events
    labels = [1] * len(fire_events) + [0] * len(neg_events)
    
    df = extract_features_for_points(ds, all_points, feature_vars)
    df['is_fire'] = labels
    
    ds.close()
    gc.collect()
    
    # ──────────────────────────────────────────────
    # 6. SPLIT TEMPORAL
    # ──────────────────────────────────────────────
    print("\n🔪 Dividiendo por años...")
    
    train_mask = df['year'] <= train_year_max
    val_mask = df['year'].isin(val_years)
    test_mask = df['year'].isin(test_years)
    
    train = df[train_mask].sample(frac=1, random_state=42).reset_index(drop=True)
    val   = df[val_mask].sample(frac=1, random_state=42).reset_index(drop=True)
    test  = df[test_mask].sample(frac=1, random_state=42).reset_index(drop=True)
    
    for name, split in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        n_fire = (split['is_fire'] == 1).sum()
        n_nofire = (split['is_fire'] == 0).sum()
        ratio = n_nofire / n_fire if n_fire > 0 else 0
        print(f"   {name}: {len(split):,} rows | ✦ {n_fire:,} fuegos | ○ {n_nofire:,} no-fuegos | ratio {ratio:.2f}")
    
    # ──────────────────────────────────────────────
    # 7. GUARDAR PARQUET
    # ──────────────────────────────────────────────
    train_path = output_path / "train.parquet"
    val_path = output_path / "val.parquet"
    test_path = output_path / "test.parquet"
    
    train.to_parquet(train_path, compression="snappy")
    val.to_parquet(val_path, compression="snappy")
    test.to_parquet(test_path, compression="snappy")
    
    print(f"\n💾 Guardado:")
    print(f"   {train_path}")
    print(f"   {val_path}")
    print(f"   {test_path}")
    
    # ──────────────────────────────────────────────
    # 8. NORMALIZAR
    # ──────────────────────────────────────────────
    print("\n📏 Normalizando features...")
    
    binary_vars = {'is_waterbody', 'is_holiday', 'is_near_fire', 'is_weekend', 'is_natura2000'}
    cols_to_norm = [c for c in feature_vars if c in train.columns and c not in binary_vars]
    
    scaler = StandardScaler()
    scaler.fit(train[cols_to_norm])
    
    scaler_path = output_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    for name, split, path in [("train", train, train_path), 
                               ("val", val, val_path), 
                               ("test", test, test_path)]:
        split_norm = split.copy()
        split_norm[cols_to_norm] = scaler.transform(split_norm[cols_to_norm])
        norm_path = str(path).replace(".parquet", "_normalized.parquet")
        split_norm.to_parquet(norm_path, compression="snappy")
        print(f"   ✅ {name} normalizado → {norm_path}")
    
    # ──────────────────────────────────────────────
    # 9. RESUMEN
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"   Features: {len(cols_to_norm)} normalizadas + {len(binary_vars & set(train.columns))} binarias")
    print(f"   Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"   Archivos en: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline tabular rápido para XGBoost")
    parser.add_argument("--datacube", type=str, 
                        default=str(PROJECT_ROOT / "data" / "IberFire.nc"))
    parser.add_argument("--output", type=str, 
                        default=str(PROJECT_ROOT / "data" / "processed" / "tabular"))
    parser.add_argument("--start_year", type=int, default=2008)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--train_max", type=int, default=2020)
    parser.add_argument("--val_years", nargs="+", type=int, default=[2021, 2022])
    parser.add_argument("--test_years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--ccaa", nargs="+", type=int, default=[12, 3, 7, 11, 1])
    args = parser.parse_args()
    
    run_fast_pipeline(
        datacube_path=args.datacube,
        output_dir=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        ccaa_target=args.ccaa,
        neg_ratio=args.neg_ratio,
        train_year_max=args.train_max,
        val_years=tuple(args.val_years),
        test_years=tuple(args.test_years),
    )
