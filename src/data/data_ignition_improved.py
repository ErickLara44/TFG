import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Variables calculadas on-the-fly desde el calendario (no existen en el datacube)
VIRTUAL_TIME_VARS = {'is_weekend', 'day_of_year_sin', 'day_of_year_cos'}

# Ruta canónica de las estadísticas pre-computadas (mean/std por canal).
# Se generan con scripts/compute_ignition_stats.py a partir del datacube.
DEFAULT_STATS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ignition_stats.json"


def load_default_stats(stats_path=None):
    """
    Carga el dict {var: {mean, std}} desde JSON.
    Si no existe, devuelve dict vacío (el caller decide el fallback).
    Genera con: python scripts/compute_ignition_stats.py
    """
    path = Path(stats_path) if stats_path is not None else DEFAULT_STATS_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_normalization_tensors(feature_vars, stats, include_fire_state=False, device=None):
    """
    Construye tensores (1, 1, C, 1, 1) de mean/std listos para normalizar un batch
    (B, T, C, H, W). Variables sin stats registradas usan mean=0, std=1 con aviso.
    Para ignición include_fire_state=False por defecto (no se añade canal extra).
    """
    means, stds, missing = [], [], []
    for v in feature_vars:
        s = stats.get(v)
        if s is None:
            missing.append(v)
            means.append(0.0)
            stds.append(1.0)
        else:
            means.append(float(s['mean']))
            std_val = float(s['std'])
            stds.append(std_val if std_val != 0.0 else 1.0)
    if include_fire_state:
        means.append(0.0)
        stds.append(1.0)
    if missing:
        print(f"⚠️ Stats faltantes para {len(missing)} vars (se usará mean=0, std=1): {missing}")
    mean_t = torch.tensor(means, dtype=torch.float32).view(1, 1, -1, 1, 1)
    std_t = torch.tensor(stds, dtype=torch.float32).view(1, 1, -1, 1, 1)
    if device is not None:
        mean_t = mean_t.to(device)
        std_t = std_t.to(device)
    return mean_t, std_t


def build_channel_stats_arrays(feature_vars, stats):
    """
    Arrays (C,) de mean/std por canal, compatibles con la API
    `PrecomputedIgnitionDataset(stats={'mean': ..., 'std': ...})`.
    """
    means, stds = [], []
    for v in feature_vars:
        s = stats.get(v)
        if s is None:
            means.append(0.0)
            stds.append(1.0)
        else:
            means.append(float(s['mean']))
            std_val = float(s['std'])
            stds.append(std_val if std_val != 0.0 else 1.0)
    return {
        'mean': torch.tensor(means, dtype=torch.float32),
        'std': torch.tensor(stds, dtype=torch.float32),
    }

DEFAULT_FEATURE_VARS = [
    # Topografía
    'elevation_mean','slope_mean',
    # Land cover base (combustible natural)
    'CLC_2018_forest_proportion','CLC_2018_scrub_proportion','CLC_2018_agricultural_proportion',
    # Land cover antrópico (fuentes de ignición humanas)
    'CLC_2018_urban_fabric_proportion','CLC_2018_artificial_proportion',
    'CLC_2018_industrial_proportion','CLC_2018_mine_proportion',
    # Contexto antrópico / barreras
    'dist_to_roads_mean','dist_to_railways_mean','dist_to_waterways_mean',
    'popdens_2020','is_waterbody','is_natura2000',
    # Meteorología - Temperatura
    't2m_mean','t2m_max','t2m_range',
    # Meteorología - Humedad
    'RH_min','RH_max','RH_range',
    # Meteorología - Viento (media + ráfagas)
    'wind_speed_mean','wind_speed_max',
    'wind_direction_mean','wind_direction_at_max_speed',
    # Meteorología - Presión y precipitación
    'surface_pressure_mean','surface_pressure_range',
    'total_precipitation_mean',
    # Vegetación / combustible vivo
    'NDVI','FAPAR','LAI',
    # Humedad del suelo (multicapa)
    'SWI_001','SWI_005','SWI_010','SWI_020',
    # Índices de riesgo y estado térmico
    'FWI','LST',
    # Señales humanas (calendario)
    'is_holiday','is_weekend','day_of_year_sin','day_of_year_cos'
]

class IgnitionDataset(Dataset):
    """
    Dataset para predicción de ignición (clasificación binaria) basado en PARCHES.
    Extrae recortes espaciales (ej. 128x128) alrededor de puntos de interés.
    
    Entrada:
      - x_seq: [T, C, H_patch, W_patch]
    Salida:
      - y: [1] (0=no fuego, 1=fuego en el centro del parche en t+1)
    """
    def __init__(self, datacube, time_indices, temporal_context=3, mode="convlstm",
                 feature_vars=None, patch_size=64, samples_per_epoch=1000,
                 balance_ratio=2.0, spatial_mask=None, max_fires_per_day=10):
        self.datacube = datacube
        self.valid_time_indices = [t['time_index'] for t in time_indices] # Lista plana de tiempos válidos
        self.temporal_context = temporal_context
        self.mode = mode
        self.feature_vars = feature_vars if feature_vars is not None else DEFAULT_FEATURE_VARS
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.balance_ratio = balance_ratio # Ratio neg/pos
        self.spatial_mask = spatial_mask  # Máscara 2D (True=Valid, False=Invalid)
        self.max_fires_per_day = max_fires_per_day

        # Pre-calcular variables virtuales de calendario (is_weekend, doy_sin, doy_cos)
        self._precompute_virtual_vars()

        # Validar variables
        self._validate_feature_vars()
        
        # Generar muestras espaciales (t, y, x)
        if not self.valid_time_indices:
            self.samples = []
        else:
            self.samples = self._generate_spatial_samples()
        
        print(f"🔥 IgnitionDataset (Patch-based) inicializado:")
        print(f"   Variables: {len(self.feature_vars)}")
        print(f"   Contexto temporal: {temporal_context} días")
        print(f"   Tamaño de parche: {patch_size}x{patch_size}")
        print(f"   Muestras totales: {len(self.samples)}")

    def _validate_feature_vars(self):
        missing = [v for v in self.feature_vars
                   if v not in self.datacube.data_vars and v not in VIRTUAL_TIME_VARS]
        if missing:
            raise ValueError(f"Variables faltantes: {missing}")

    def _precompute_virtual_vars(self):
        """Series 1D (len=T_total) de variables derivadas del calendario."""
        times = pd.to_datetime(self.datacube.time.values)
        doy = times.dayofyear.values.astype(np.float32)
        two_pi = 2.0 * np.pi
        self._virtual_cache = {
            'is_weekend': (times.dayofweek.values >= 5).astype(np.float32),
            'day_of_year_sin': np.sin(two_pi * doy / 365.25).astype(np.float32),
            'day_of_year_cos': np.cos(two_pi * doy / 365.25).astype(np.float32),
        }

    def _generate_spatial_samples(self):
        """
        Genera una lista de muestras (t, y, x) balanceadas.
        Estrategia:
        1. Identificar píxeles con fuego en los tiempos válidos.
        2. Muestrear aleatoriamente píxeles sin fuego.
        """
        print("📍 Generando muestras espaciales...")
        fire_samples = []
        # Set de (t_target, y, x) con fuego REAL (sin filtro físico) para rechazar negativos.
        # Clave: spec "no-fire instances que no interfieran con fire instances".
        raw_fire_coords = set()

        # 1. Encontrar fuegos (esto puede ser lento si se hace naive, optimizamos)
        # Iteramos por los tiempos válidos y buscamos dónde hay fuego
        # Para no tardar una eternidad, hacemos un muestreo inteligente o usamos pre-conocimiento
        # Aquí iteraremos buscando máscaras de fuego no vacías
        
        # Optimizacion: Leer 'is_fire' solo para los tiempos necesarios en chunks
        # Pero para simplificar y asegurar que encontramos fuegos, usaremos un enfoque heurístico
        # o iteraremos rápido si el dataset cabe en memoria (no cabe).
        
        # Optimizacion: Leer en chunks para reducir I/O
        chunk_size = 100
        valid_indices_sorted = sorted(self.valid_time_indices)
        
        print(f"   🔍 Buscando puntos de ignición activos (Chunked scan)...")
        
        for i in tqdm(range(0, len(valid_indices_sorted), chunk_size), desc="Scanning chunks"):
            chunk_indices = valid_indices_sorted[i:i+chunk_size]
            if not chunk_indices:
                continue
                
            t_min = min(chunk_indices)
            t_max = max(chunk_indices)
            
            # Leer rango [t_min+1, t_max+2)
            t_start_read = t_min + 1
            t_end_read = min(t_max + 2, len(self.datacube.time))
            
            if t_start_read >= len(self.datacube.time):
                continue
                
            # Cargar chunk de datos (FUEGO + FILTROS FÍSICOS)
            # OPTIMIZACIÓN: Cargar todo de golpe para evitar lecturas lentas en bucle
            fire_data_chunk = self.datacube["is_fire"].isel(time=slice(t_start_read, t_end_read)).values
            
            # Cargar variables de filtro si existen (evita lazy loading repetido)
            fwi_chunk = None
            if 'FWI' in self.datacube:
                 fwi_chunk = self.datacube["FWI"].isel(time=slice(t_start_read, t_end_read)).values
                 
            swi_chunk = None
            if 'SWI_001' in self.datacube:
                 swi_chunk = self.datacube["SWI_001"].isel(time=slice(t_start_read, t_end_read)).values
                 
            rh_chunk = None
            if 'RH_min' in self.datacube:
                 rh_chunk = self.datacube["RH_min"].isel(time=slice(t_start_read, t_end_read)).values

            for t in chunk_indices:
                t_target = t + 1
                if t_target >= len(self.datacube.time):
                    continue
                
                # Índice relativo en el chunk
                local_idx = t_target - t_start_read
                if 0 <= local_idx < fire_data_chunk.shape[0]:
                    
                    # 1. Máscara de Fuego Crudo
                    fire_mask = fire_data_chunk[local_idx] > 0

                    # Registrar TODAS las coordenadas con fuego real (sin filtro físico)
                    # para rechazar negativos que solapen con cualquier fuego.
                    ys_raw, xs_raw = np.where(fire_mask)
                    for yy, xx in zip(ys_raw, xs_raw):
                        raw_fire_coords.add((t_target, int(yy), int(xx)))

                    # 2. Aplicar Filtros Físicos Vectorizados
                    if np.sum(fire_mask) > 0:
                        physics_mask = np.ones_like(fire_mask, dtype=bool)
                        
                        # Filtro FWI >= 1.0 (Si hay FWI, el fuego debe tener riesgo mínimo)
                        if fwi_chunk is not None:
                            physics_mask &= (fwi_chunk[local_idx] >= 1.0)
                            
                        # Filtro Suelo Empapado SWI <= 80%
                        if swi_chunk is not None:
                            physics_mask &= (swi_chunk[local_idx] <= 80.0)
                            
                        # Filtro Humedad Extrema RH <= 90%
                        if rh_chunk is not None:
                             physics_mask &= (rh_chunk[local_idx] <= 90.0)
                             
                        # Combina Fuego Real + Física Posible
                        valid_fire_mask = fire_mask & physics_mask
                        
                        # Extraer coordenadas válidas
                        ys, xs = np.where(valid_fire_mask)
                        n_fires = len(ys)
                        
                        if n_fires > 0:
                            # Submuestreo (Max fires per day)
                            limit = self.max_fires_per_day
                            if limit is None:
                                n_sample = n_fires
                            else:
                                n_sample = min(n_fires, limit)

                            indices = np.random.choice(n_fires, size=n_sample, replace=False)
                            for idx in indices:
                                # Comprobar máscara espacial (Zona de interés)
                                if self.spatial_mask is not None:
                                     if not self.spatial_mask[ys[idx], xs[idx]]:
                                         continue
                                
                                fire_samples.append({
                                    "time_index": t,
                                    "y": ys[idx],
                                    "x": xs[idx],
                                    "label": 1.0
                                })

        
        n_fire = len(fire_samples)
        print(f"   🔥 Encontrados {n_fire} puntos de ignición.")
        
        if n_fire == 0:
            print("   ⚠️ No se encontraron fuegos. Generando muestras aleatorias.")
            # Fallback a random sampling
            n_fire = 100
        
        # 2. Generar muestras negativas (No fuego)
        # Muestreamos aleatoriamente del espacio-tiempo válido
        n_neg = int(n_fire * self.balance_ratio)
        # Si samples_per_epoch está definido, ajustamos
        if self.samples_per_epoch > 0:
            # Intentar respetar el total solicitado manteniendo el ratio
            # O simplemente llenar hasta samples_per_epoch
            pass 

        no_fire_samples = []
        H, W = self.datacube.sizes['y'], self.datacube.sizes['x']
        
        # Mapa de validez para sampling negativo
        valid_map = np.ones((H, W), dtype=bool)
        if self.spatial_mask is not None:
             if self.spatial_mask.shape == (H, W):
                 valid_map = valid_map & self.spatial_mask
        
        valid_y, valid_x = np.where(valid_map)
        if len(valid_y) == 0:
            # Fallback
            valid_y, valid_x = np.where(np.ones((H, W), dtype=bool))
        n_valid = len(valid_y)

        # Rechazo por solape: reintenta si el candidato coincide con un fuego real.
        max_attempts_per_neg = 50
        rejected_overlap = 0
        for _ in range(n_neg):
            accepted = False
            for _attempt in range(max_attempts_per_neg):
                t = int(np.random.choice(self.valid_time_indices))
                idx_pixel = np.random.randint(0, n_valid)
                y = int(valid_y[idx_pixel])
                x = int(valid_x[idx_pixel])
                t_target = t + 1

                if (t_target, y, x) in raw_fire_coords:
                    rejected_overlap += 1
                    continue

                no_fire_samples.append({
                    "time_index": t,
                    "y": y,
                    "x": x,
                    "label": 0.0
                })
                accepted = True
                break

            if not accepted:
                # Tras N reintentos, aceptamos el último candidato (eventos raros)
                no_fire_samples.append({
                    "time_index": t,
                    "y": y,
                    "x": x,
                    "label": 0.0
                })

        if rejected_overlap > 0:
            print(f"   🚫 {rejected_overlap} candidatos negativos rechazados por solape con fuego real.")
            
        # Combinar
        all_samples = fire_samples + no_fire_samples
        np.random.shuffle(all_samples)
        
        # Recortar al límite solicitado si es necesario
        if self.samples_per_epoch is not None and len(all_samples) > self.samples_per_epoch:
            all_samples = all_samples[:self.samples_per_epoch]
            
        return all_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t0 = sample["time_index"]
        cy, cx = sample["y"], sample["x"] # Centro del parche
        
        # Calcular límites del parche
        half_size = self.patch_size // 2
        y_start = max(0, cy - half_size)
        y_end = min(self.datacube.sizes['y'], cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(self.datacube.sizes['x'], cx + half_size)
        
        # Ajustar si el parche es más pequeño que patch_size (bordes)
        # Simplemente aceptamos parches más pequeños o hacemos padding luego.
        # Aquí haremos padding simple con edge replication si es necesario, 
        # o mejor: movemos la ventana para que quepa siempre si es posible.
        
        # Estrategia: Mover ventana para que sea siempre patch_size (si el mapa es > patch_size)
        if y_end - y_start < self.patch_size:
            if y_start == 0: y_end = min(self.datacube.sizes['y'], self.patch_size)
            if y_end == self.datacube.sizes['y']: y_start = max(0, self.datacube.sizes['y'] - self.patch_size)
            
        if x_end - x_start < self.patch_size:
            if x_start == 0: x_end = min(self.datacube.sizes['x'], self.patch_size)
            if x_end == self.datacube.sizes['x']: x_start = max(0, self.datacube.sizes['x'] - self.patch_size)

        # Rango temporal
        t_start = t0 - (self.temporal_context - 1)
        t_end = t0 + 1
        
        # Padding temporal si es necesario
        pad_time_left = 0
        if t_start < 0:
            pad_time_left = abs(t_start)
            t_start = 0

        # Leer datos
        H_patch = y_end - y_start
        W_patch = x_end - x_start
        x_seq = []
        for var in self.feature_vars:
            if var in VIRTUAL_TIME_VARS:
                # Serie escalar por time → broadcast espacial a (T_read, H, W)
                vals = self._virtual_cache[var][t_start:t_end]
                arr = np.broadcast_to(
                    vals[:, None, None], (len(vals), H_patch, W_patch)
                ).astype(np.float32).copy()
                if pad_time_left > 0:
                    first_frame = arr[0:1]
                    padding = np.repeat(first_frame, pad_time_left, axis=0)
                    arr = np.concatenate([padding, arr], axis=0)
            elif "time" in self.datacube[var].dims:
                # Slice temporal y espacial: (T, H_patch, W_patch)
                arr = self.datacube[var].isel(
                    time=slice(t_start, t_end),
                    y=slice(y_start, y_end),
                    x=slice(x_start, x_end)
                ).values.astype(np.float32)

                # Padding temporal
                if pad_time_left > 0:
                    first_frame = arr[0:1]
                    padding = np.repeat(first_frame, pad_time_left, axis=0)
                    arr = np.concatenate([padding, arr], axis=0)
            else:
                # Estático: (H_patch, W_patch)
                arr_static = self.datacube[var].isel(
                    y=slice(y_start, y_end),
                    x=slice(x_start, x_end)
                ).values.astype(np.float32)
                # Repetir temporalmente
                arr = np.repeat(arr_static[np.newaxis, ...], self.temporal_context, axis=0)

            x_seq.append(arr)
            
        # Stack: (T, C, H, W)
        x_seq = np.stack(x_seq, axis=1)
        
        # Limpiar NaNs en la entrada
        if np.isnan(x_seq).any():
            x_seq = np.nan_to_num(x_seq, nan=0.0)
        
        # Etiqueta
        y_label = torch.tensor([sample["label"]], dtype=torch.float32)
        
        # Ajustar formato
        if self.mode == "cnn":
            T, C, H, W = x_seq.shape
            x_out = x_seq.reshape(T*C, H, W)
        else:
            x_out = x_seq

        return torch.FloatTensor(x_out), y_label

    def get_feature_info(self):
        """Información sobre las variables de entrada"""
        print("🔍 INFORMACIÓN DE VARIABLES:")
        for i, var in enumerate(self.feature_vars):
            print(f"   Canal {i:2d}: {var}")
        
        return self.feature_vars

    def get_class_distribution(self, n_samples=None):
        """Analiza la distribución de clases en el dataset"""
        if n_samples is None:
            n_samples = len(self.samples)
        else:
            n_samples = min(n_samples, len(self.samples))
        
        fire_count = 0
        no_fire_count = 0
        
        print(f"📊 Analizando distribución de clases en {n_samples} muestras...")
        
        for i in range(n_samples):
            _, y = self[i]
            if y.item() > 0.5:
                fire_count += 1
            else:
                no_fire_count += 1
        
        total = fire_count + no_fire_count
        fire_pct = (fire_count / total) * 100 if total > 0 else 0
        no_fire_pct = (no_fire_count / total) * 100 if total > 0 else 0
        
        print(f"   🔥 Con fuego: {fire_count} ({fire_pct:.1f}%)")
        print(f"   💧 Sin fuego: {no_fire_count} ({no_fire_pct:.1f}%)")
        
        if abs(fire_pct - 50) > 20:  # Si está muy desbalanceado
            print(f"   ⚠️ Dataset desbalanceado. Considera usar balance_classes=True")
        
        return {
            'fire_samples': fire_count,
            'no_fire_samples': no_fire_count,
            'fire_percentage': fire_pct,
            'balance_ratio': min(fire_count, no_fire_count) / max(fire_count, no_fire_count)
        }

    def get_sample_stats(self, n_samples=100):
        """Estadísticas generales del dataset"""
        n_samples = min(n_samples, len(self.samples))

        shapes = []
        fire_areas = []

        print(f"📊 Analizando {n_samples} muestras...")

        for i in range(n_samples):
            x, y = self[i]
            shapes.append(x.shape)

            # Para ignición, y es escalar, pero podemos obtener el área del mapa original
            sample_info = self.samples[i]
            t_next = sample_info["time_index"] + 1
            if t_next < len(self.datacube.time):
                y_map = self.datacube["is_fire"].isel(time=t_next).values
                fire_areas.append(np.sum(y_map))
        
        print(f"   📐 Shape de entrada: {shapes[0]}")
        print(f"   🔥 Área promedio de fuego: {np.mean(fire_areas):.1f} píxeles")
        print(f"   🔥 Área máxima de fuego: {np.max(fire_areas):.0f} píxeles")
        
        return {
            'input_shape': shapes[0],
            'avg_fire_area': np.mean(fire_areas),
            'max_fire_area': np.max(fire_areas)
        }


class PrecomputedIgnitionDataset(Dataset):
    """
    Dataset optimizado que lee parches pre-calculados desde disco (.pt).
    Evita la lectura lenta del NetCDF durante el entrenamiento.
    """
    def __init__(self, patches_dir, indices, mode="convlstm", stats=None, augment=False,
                 return_meta=False):
        self.patches_dir = Path(patches_dir)
        self.indices = indices
        self.mode = mode
        self.stats = stats
        self.augment = augment
        self.return_meta = return_meta

        # Verificar que existen los archivos
        if not self.patches_dir.exists():
            raise FileNotFoundError(f"No se encontró el directorio de parches: {patches_dir}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Usar el índice mapeado real en lugar del índice 0..N
        real_idx = self.indices[idx]
        file_path = self.patches_dir / f"patch_{real_idx}.pt"

        data = torch.load(file_path)
        x = data['x']  # (T, C, H, W)
        y = data['y']

        # 1. Manejo de NaNs (Crítico)
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        # 2. Normalización
        if self.stats is not None:
            # stats['mean'] es (C,) -> expandir a (1, C, 1, 1)
            mean = torch.as_tensor(self.stats['mean'], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            std = torch.as_tensor(self.stats['std'], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            x = (x - mean) / std

        # 3. Data Augmentation
        if self.augment:
            import random
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-1])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[-2])
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[-2, -1])

        # Ajustar dimensiones según modo
        if self.mode == "cnn":
            T, C, H, W = x.shape
            x = x.reshape(T * C, H, W)

        if self.return_meta:
            meta = {
                'time_index': int(data.get('time_index', -1)),
                'cy': int(data.get('cy', -1)),
                'cx': int(data.get('cx', -1)),
            }
            return x, y, meta

        return x, y

def precompute_patches(dataset, output_dir, start_index=0):
    """
    Genera y guarda los parches en disco para entrenamiento rápido.

    Cada .pt contiene:
        {'x': features, 'y': label, 'time_index': int, 'cy': int, 'cx': int}
    La metadata (time_index, cy, cx) permite proyectar predicciones en el mapa
    durante evaluación.

    Args:
        dataset (IgnitionDataset): Dataset original (lento)
        output_dir (str): Directorio donde guardar los .pt
        start_index (int): Índice inicial para nombrar los archivos (utile para chunking)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Pre-computando {len(dataset)} parches en {output_dir} (Start ID: {start_index})...")

    saved_count = 0
    for i in tqdm(range(len(dataset))):
        try:
            x, y = dataset[i]
            meta = dataset.samples[i]

            # Guardar como diccionario comprimido
            # Usamos start_index + i para evitar colisiones entre chunks
            global_idx = start_index + i
            save_path = output_path / f"patch_{global_idx}.pt"

            # Si ya existe, saltar (Resume capability)
            if save_path.exists():
                saved_count += 1
                continue

            torch.save({
                'x': x.clone(),
                'y': y.clone(),
                'time_index': int(meta['time_index']),
                'cy': int(meta['y']),
                'cx': int(meta['x']),
            }, save_path)
            saved_count += 1

        except Exception as e:
            print(f"❌ Error en índice {i}: {e}")

    print("✅ Pre-computación completada.")
    return saved_count


# Ratios por defecto: 2:1 neg/pos en train/val/test (benchmark-style).
# Para test "deployment" (distribución cercana a la real), pasar un ratio alto
# (ej. {'test_deployment': 100.0}) como split adicional.
DEFAULT_BALANCE_RATIOS = {'train': 2.0, 'val': 2.0, 'test': 2.0}


def create_ignition_datasets(datacube, splits, temporal_context=3,
                             patch_size=64, samples_train=2000, samples_val=500,
                             samples_test=None, balance_ratios=None,
                             spatial_masks=None, max_fires_per_day=10):
    """
    Crea datasets de ignición basados en parches.

    Args:
        splits (dict): {'train': [...], 'val': [...], 'test': [...], ...}
            Puede incluir claves extra como 'test_deployment' para un test con
            distribución cercana a la real.
        balance_ratios (dict): ratio neg/pos por split. Default: 2.0 para
            train/val/test. Pasar ratio alto (p.ej. 100.0) para el split
            'test_deployment' si se quiere evaluar en distribución realista.
        samples_test (int|None): cap de muestras para el split 'test'. Si es
            None, usa `samples_val`.
    """
    print("🔥 CREANDO DATASETS DE IGNICIÓN (PATCH-BASED)...")

    if balance_ratios is None:
        balance_ratios = dict(DEFAULT_BALANCE_RATIOS)
    spatial_masks = spatial_masks or {}

    samples_per_split = {
        'train': samples_train,
        'val': samples_val,
        'test': samples_test if samples_test is not None else samples_val,
    }

    datasets = {}
    for split_name, indices in splits.items():
        if not indices:
            continue
        ratio = balance_ratios.get(split_name, 2.0)
        n_samples = samples_per_split.get(split_name, samples_val)
        print(f"   📦 Split '{split_name}': ratio neg/pos={ratio}, samples={n_samples}")
        datasets[split_name] = IgnitionDataset(
            datacube, indices,
            temporal_context=temporal_context,
            patch_size=patch_size,
            samples_per_epoch=n_samples,
            balance_ratio=ratio,
            spatial_mask=spatial_masks.get(split_name),
            max_fires_per_day=max_fires_per_day,
        )

    return datasets

def create_precomputed_ignition_datasets(datacube, splits, output_dir="data/processed/patches",
                                         temporal_context=3, patch_size=64,
                                         samples_train=2000, samples_val=500,
                                         samples_test=None, balance_ratios=None,
                                         spatial_masks=None, start_indices=None):
    """
    Versión optimizada que pre-calcula parches antes de entrenar.
    Args:
        start_indices (dict): {'train': 0, 'val': 0, ...} para continuar numeración.
    """
    # 1. Crear datasets originales (lentos)
    datasets = create_ignition_datasets(
        datacube, splits, temporal_context,
        patch_size=patch_size,
        samples_train=samples_train,
        samples_val=samples_val,
        samples_test=samples_test,
        balance_ratios=balance_ratios,
        spatial_masks=spatial_masks,
    )

    if start_indices is None:
        start_indices = {k: 0 for k in datasets.keys()}
    
    # 2. Pre-calcular y guardar
    precomputed_datasets = {}
    
    for split_name, dataset in datasets.items():
        split_dir = Path(output_dir) / split_name
        
        # Si ya existen, saltar o continuar
        output_path = Path(split_dir)
        output_path.mkdir(parents=True, exist_ok=True)
              
        current_start_idx = start_indices.get(split_name, 0)
        precompute_patches(dataset, split_dir, start_index=current_start_idx)
            
        # 3. Crear datasets rápidos (esto solo referencia lo que hay en disco)
        # OJO: Si estamos en chunking loop, esto retornará un dataset parcial o incoherente
        # hasta que termine todo. Pero es útil para testear.
        # Asumimos que PrecomputedIgnitionDataset escaneará todo el dir.
        
        # Contamos cuántos archivos hay realmente
        files = list(output_path.glob("patch_*.pt"))
        precomputed_datasets[split_name] = PrecomputedIgnitionDataset( # Asumiendo que existe o es dummy
            split_dir, 
            indices=list(range(len(files)))
        )
        
    return precomputed_datasets
