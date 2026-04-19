import json
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import hashlib
from tqdm import tqdm
from src.data.data_ignition_improved import IgnitionDataset

# Ruta canónica de las estadísticas pre-computadas (mean/std por canal).
# Se generan con scripts/compute_spread_stats.py a partir del datacube.
DEFAULT_STATS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "spread_stats.json"

def load_default_stats(stats_path=None):
    """
    Carga el dict {var: {mean, std}} desde JSON.
    Si no existe, devuelve dict vacío (el caller decide el fallback).
    Genera con: python scripts/compute_spread_stats.py
    """
    path = Path(stats_path) if stats_path is not None else DEFAULT_STATS_PATH
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def get_wind_indices(feature_vars):
    """
    Devuelve los índices de wind_u y wind_v dentro de feature_vars.
    Usado por las augmentaciones que rotan/flipean el vector viento.
    Retorna (None, None) si alguna de las dos no está en la lista.
    """
    try:
        idx_u = feature_vars.index('wind_u')
    except ValueError:
        idx_u = None
    try:
        idx_v = feature_vars.index('wind_v')
    except ValueError:
        idx_v = None
    return idx_u, idx_v


def build_normalization_tensors(feature_vars, stats, include_fire_state=True, device=None):
    """
    Construye tensores (1, 1, C, 1, 1) de mean/std listos para normalizar un batch
    (B, T, C, H, W). Canal extra de is_fire no se normaliza (mean=0, std=1).
    Variables sin stats registradas usan mean=0, std=1 con aviso.
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


# 🔹 Variables por defecto optimizadas para la física del spread
# NOTA: popdens y dist_to_roads se eliminan (señal de ignición humana, no de propagación)
# NOTA: CLC_current_* mapea dinámicamente al año correcto (2006/2012/2018)
# NOTA: wind_u, wind_v son componentes cartesianas del viento (evitan discontinuidad en 0°/360°)
DEFAULT_FEATURE_VARS = [
    # Topografía
    'elevation_mean','slope_mean',
    # Orientación solar (aspect sur: 4=SE, 5=S, 6=SW → combustible más seco)
    'aspect_4','aspect_5','aspect_6',
    # Combustible (CLC dinámico por año, corrige usar CLC_2018 en fuegos antiguos)
    'CLC_current_forest_proportion','CLC_current_scrub_proportion','CLC_current_agricultural_proportion',
    # Barreras físicas
    'dist_to_waterways_mean','is_waterbody',
    # Meteo - Temperatura
    't2m_mean','t2m_max',
    # Meteo - Humedad
    'RH_min','RH_range',
    # Meteo - Viento (media + ráfagas + vector cartesiano)
    'wind_speed_mean','wind_speed_max',
    'wind_direction_mean','wind_direction_at_max_speed',
    'wind_u','wind_v',
    # Meteo - Precipitación e índice de riesgo
    'total_precipitation_mean','FWI',
    # Combustible vivo
    'NDVI','FAPAR','LAI',
    # Humedad del suelo (superficial + raíz)
    'SWI_001','SWI_010',
    # Estado térmico superficial
    'LST'
]

class SpreadDataset(Dataset):
    """
    Dataset para predicción de propagación espacial.
    🔥 Incluye estado actual del fuego como contexto esencial
    """
    def __init__(self, datacube, indices, temporal_context=3, include_fire_state=True, 
                 filter_fire_samples=True, min_fire_pixels=5, feature_vars=None,
                 cache_dir="data/processed/cache_spread", preload_ram=False, crop_size=224):
        self.datacube = datacube
        self.indices = indices
        self.temporal_context = temporal_context
        self.include_fire_state = include_fire_state
        self.filter_fire_samples = filter_fire_samples
        self.min_fire_pixels = min_fire_pixels
        self.preload_ram = preload_ram
        self.crop_size = crop_size

        # ✅ Usar tus variables fijas
        self.feature_vars = feature_vars if feature_vars is not None else DEFAULT_FEATURE_VARS

        # Validar que existen en el datacube
        # Excluir variables virtuales (calculadas on-the-fly) de la validación
        virtual_prefixes = ['CLC_current_']
        virtual_exact = ['wind_u', 'wind_v', 'hydric_stress', 'solar_risk']
        
        feature_vars_to_check = []
        for v in self.feature_vars:
            is_virtual = (v in virtual_exact) or any(v.startswith(p) for p in virtual_prefixes)
            if not is_virtual:
                feature_vars_to_check.append(v)

        missing = [v for v in feature_vars_to_check if v not in datacube.data_vars]
        if missing:
            raise ValueError(f"❌ Variables no encontradas en datacube: {missing}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Filtrar muestras que tienen fuego actual (para propagación)
        if filter_fire_samples:
            self.indices = self._load_or_filter_samples()
        
        print(f"🔥 SpreadDataset inicializado:")
        print(f"   Variables: {len(self.feature_vars)} + {'1 (fire_state)' if include_fire_state else '0'}")
        print(f"   Contexto temporal: {temporal_context} días")
        print(f"   Muestras totales: {len(self.indices)}")
        print(f"   Filtrado por fuego: {filter_fire_samples}")

        self.ram_cache = []
        if self.preload_ram:
            self._preload_to_ram()

    def _load_or_filter_samples(self):
        """Carga índices filtrados desde caché o los genera"""
        # Hash basado en los índices de entrada
        indices_str = str(sorted([x['time_index'] for x in self.indices]))
        split_hash = hashlib.md5(indices_str.encode()).hexdigest()[:10]
        cache_file = self.cache_dir / f"spread_indices_{split_hash}.pt"
        
        if cache_file.exists():
            print(f"📦 Cargando índices de propagación desde caché: {cache_file}")
            return torch.load(cache_file, weights_only=False)
            
        print("📍 Filtrando muestras con fuego (Cache miss)...")
        indices = self._filter_samples_with_fire_logic()
        
        print(f"💾 Guardando índices en caché: {cache_file}")
        torch.save(indices, cache_file)
        return indices

    def _filter_samples_with_fire_logic(self):
        """Filtra muestras que tienen fuego en al menos uno de los timesteps de contexto (Vectorizado)"""
        print("⚡ Pre-calculando actividad de fuego en todo el dataset (esto puede tardar unos segundos)...")
        # Calcular suma de pixeles de fuego por timestep de una sola vez
        # self.datacube['is_fire'] es (Time, Y, X). Sumamos sobre Y, X -> (Time,)
        fire_sums = self.datacube['is_fire'].sum(dim=['y', 'x']).values
        
        # Boolean array: ¿Hay fuego significativo en tiempo t?
        has_fire_per_time = (fire_sums >= self.min_fire_pixels)
        
        valid_indices = []
        # Iterar sobre los índices candidatos
        for sample_info in tqdm(self.indices, desc="Filtering samples"):
            t0 = sample_info["time_index"]
            
            # Verificar si CUALQUIERA de los steps de contexto (t0, t0-1, t0-2) tiene fuego
            # Definir rango de tiempos para este sample
            # t va desde t0-(ctx-1) hasta t0
            t_start = t0 - (self.temporal_context - 1)
            t_end = t0 + 1 # exclusive for slice
            
            # Clamp indices
            t_start = max(0, t_start)
            t_end = min(len(has_fire_per_time), t_end)
            
            # Check slice
            if np.any(has_fire_per_time[t_start:t_end]):
                valid_indices.append(sample_info)
        
        print(f"   📊 Muestras con fuego: {len(valid_indices)}/{len(self.indices)} "
              f"({len(valid_indices)/len(self.indices)*100:.1f}%)")
        return valid_indices

    def _preload_to_ram(self):
        """Pre-carga todos los tensores en RAM para entrenamiento ultra-rápido"""
        print(f"🚀 Pre-cargando {len(self.indices)} muestras en RAM...")
        self.ram_cache = []
        for i in tqdm(range(len(self.indices)), desc="Loading to RAM"):
            self.ram_cache.append(self._load_sample(i))
        print(f"✅ Pre-carga completada. RAM lista.")

    def __len__(self):
        return len(self.indices)

    def _load_sample(self, idx):
        sample_info = self.indices[idx]
        t0 = sample_info["time_index"]
        
        # 1. Determinar Centro del Crop (Leemos solo la mascara de fuego actual)
        # Esto es rápido (1 solo canal boolean/int8)
        # t0 es el último timestamp del contexto de entrada
        try:
             # isel supports lazy loading if backed by proper backend, .values triggers read
            fire_mask_full = self.datacube["is_fire"].isel(time=t0).values
        except Exception:
            # Fallback for safe access
            fire_mask_full = np.zeros((self.datacube.sizes['y'], self.datacube.sizes['x']), dtype=np.float32)

        if self.crop_size is not None:
            cy, cx = self._get_fire_center(fire_mask_full)
        else:
            cy, cx = fire_mask_full.shape[0]//2, fire_mask_full.shape[1]//2

        # 2. Calcular slices de lectura (optimizados para no leer todo el disco/memoria)
        if self.crop_size is not None:
            H_full, W_full = self.datacube.sizes['y'], self.datacube.sizes['x']
            r = self.crop_size // 2
            
            y1_req, y2_req = cy - r, cy + r
            x1_req, x2_req = cx - r, cx + r
            
            # Intersección con la imagen válida (lo que podemos leer)
            y1_sel, y2_sel = max(0, y1_req), min(H_full, y2_req)
            x1_sel, x2_sel = max(0, x1_req), min(W_full, x2_req)
            
            # Padding necesario (si el crop se sale)
            pad_top = max(0, -y1_req)
            pad_bottom = max(0, y2_req - H_full)
            pad_left = max(0, -x1_req)
            pad_right = max(0, x2_req - W_full)
            
            # Dimensiones esperadas del crop leído
            h_read = y2_sel - y1_sel
            w_read = x2_sel - x1_sel
        
        # --- 3. Leer secuencia temporal (solo el crop) ---
        x_seq = []
        for dt in range(self.temporal_context):
            t = t0 - (self.temporal_context - 1 - dt)
            t = max(0, min(t, len(self.datacube.time) - 1))  # clamp

            channels = []
            for var in self.feature_vars:
                # Usar método inteligente que calcula variables derivadas si no existen
                arr_crop = self._read_variable_smart(var, t, y1_sel, y2_sel, x1_sel, x2_sel)
                
                # Aplicar padding si es necesario
                if self.crop_size is not None and (pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0 or h_read < self.crop_size or w_read < self.crop_size):
                     if h_read == 0 or w_read == 0:
                         arr_padded = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
                     else:
                         arr_padded = np.pad(arr_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                     channels.append(arr_padded.astype(np.float32))
                else:
                     channels.append(arr_crop.astype(np.float32))

            # Estado del fuego
            if self.include_fire_state:
                if self.crop_size is not None:
                    fire_crop = self.datacube["is_fire"].isel(time=t, y=slice(y1_sel, y2_sel), x=slice(x1_sel, x2_sel)).values
                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0 or h_read < self.crop_size or w_read < self.crop_size:
                        if h_read == 0 or w_read == 0:
                            fire_padded = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
                        else:
                            fire_padded = np.pad(fire_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                        channels.append(fire_padded.astype(np.float32))
                    else:
                        channels.append(fire_crop.astype(np.float32))
                else:
                    channels.append(self.datacube["is_fire"].isel(time=t).values.astype(np.float32))

            x_seq.append(np.stack(channels, axis=0))

        x_seq = np.stack(x_seq, axis=0)  # (T, C, H, W)

        # --- 4. Leer Label (t+1) ---
        t_next = min(t0 + 1, len(self.datacube.time) - 1)
        if self.crop_size is not None:
             y_crop = self.datacube["is_fire"].isel(time=t_next, y=slice(y1_sel, y2_sel), x=slice(x1_sel, x2_sel)).values
             if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0 or h_read < self.crop_size or w_read < self.crop_size:
                 if h_read == 0 or w_read == 0:
                     y_padded = np.zeros((self.crop_size, self.crop_size), dtype=np.float32)
                 else:
                     y_padded = np.pad(y_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                 y_map = y_padded.astype(np.float32)
             else:
                 y_map = y_crop.astype(np.float32)
        else:
             y_map = self.datacube["is_fire"].isel(time=t_next).values.astype(np.float32)

        y_map = np.expand_dims(y_map, axis=0)
        
        # Limpiar NaNs
        if np.isnan(x_seq).any(): x_seq = np.nan_to_num(x_seq, nan=0.0)
        
        return torch.FloatTensor(x_seq), torch.FloatTensor(y_map)

    def _read_variable_smart(self, var, t, y1, y2, x1, x2):
        """Lee una variable del datacube, calculándola si es derivada (Viento, CLC, Estrés)"""
        
        # --- 1. Variables Directas ---
        if var in self.datacube:
            if 'time' in self.datacube[var].dims:
                if self.crop_size is not None:
                    return self.datacube[var].isel(time=t, y=slice(y1, y2), x=slice(x1, x2)).values
                else:
                    return self.datacube[var].isel(time=t).values
            else:
                if self.crop_size is not None:
                    return self.datacube[var].isel(y=slice(y1, y2), x=slice(x1, x2)).values
                else:
                    return self.datacube[var].values

        # --- 2. Variables Derivadas (On-the-fly) ---
        
        # 💨 Vector Viento
        if var == 'wind_u' or var == 'wind_v':
            ws = self._read_variable_smart('wind_speed_mean', t, y1, y2, x1, x2)
            wd = self._read_variable_smart('wind_direction_mean', t, y1, y2, x1, x2)
            wd_rad = np.deg2rad(wd)
            if var == 'wind_u':
                return (ws * np.sin(wd_rad)).astype(np.float32)
            else: # wind_v
                return (ws * np.cos(wd_rad)).astype(np.float32)
                
        # 🌡️ Estrés Hídrico
        if var == 'hydric_stress':
            t2m = self._read_variable_smart('t2m_mean', t, y1, y2, x1, x2)
            lst = self._read_variable_smart('LST', t, y1, y2, x1, x2)
            # Heurística simple K vs C
            if np.nanmean(lst) > 200 and np.nanmean(t2m) < 100:
                return (lst - (t2m + 273.15)).astype(np.float32)
            else:
                return (lst - t2m).astype(np.float32)

        # ☀️ Riesgo Solar
        if var == 'solar_risk':
             t2m_max = self._read_variable_smart('t2m_max', t, y1, y2, x1, x2)
             # Aproximación: Usar aspect_4, 5, 6 (Sur)
             south_aspect = np.zeros_like(t2m_max)
             for a in ['aspect_4', 'aspect_5', 'aspect_6']:
                 if a in self.datacube:
                     south_aspect += self._read_variable_smart(a, t, y1, y2, x1, x2)
             return (t2m_max * south_aspect).astype(np.float32)
             
        # 🌿 Dynamic CLC
        if var.startswith('CLC_current_'):
            # Formato esperado: CLC_current_forest_proportion
            # Extraer sufijo
            suffix = var.replace('CLC_current_', '')
            
            # Determinar año
            year = pd.to_datetime(self.datacube.time.values[t]).year
            
            if year <= 2009:
                target_var = f"CLC_2006_{suffix}"
            elif year <= 2015:
                target_var = f"CLC_2012_{suffix}"
            else:
                target_var = f"CLC_2018_{suffix}"
                
            # Intentar leer la variable mapeada
            if target_var in self.datacube:
                return self._read_variable_smart(target_var, t, y1, y2, x1, x2)
            else:
                # Fallback a 2018 si no existe la específica
                fallback = f"CLC_2018_{suffix}"
                if fallback in self.datacube:
                     return self._read_variable_smart(fallback, t, y1, y2, x1, x2)
                
        # No encontrado
        # Retornar ceros para no romper el flujo
        # print(f"⚠️ Variable desconocida: {var}")
        H = y2 - y1 if self.crop_size is not None else self.datacube.sizes['y']
        W = x2 - x1 if self.crop_size is not None else self.datacube.sizes['x']
        return np.zeros((H, W), dtype=np.float32)

    def _get_fire_center(self, fire_mask):
        """Encuentra el centroide del fuego o devuelve el centro de la imagen si no hay fuego"""
        y_idxs, x_idxs = np.where(fire_mask > 0)
        if len(y_idxs) > 0:
            return int(np.mean(y_idxs)), int(np.mean(x_idxs))
        
        # Si no hay fuego, retornar centro del mapa
        return fire_mask.shape[0] // 2, fire_mask.shape[1] // 2

    def _crop_array(self, arr, cy, cx):
        """Recorta un array (..., H, W) centrado en (cy, cx) con padding si es necesario"""
        # arr shape: (..., H, W)
        H, W = arr.shape[-2:]
        r = self.crop_size // 2
        
        y1 = cy - r
        x1 = cx - r
        y2 = cy + r
        x2 = cx + r
        
        # Padding si nos salimos
        pad_top = max(0, -y1)
        pad_left = max(0, -x1)
        pad_bottom = max(0, y2 - H)
        pad_right = max(0, x2 - W)
        
        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            # Pad simple con ceros
            # array np.pad usa ((bef, aft), (bef, aft)...) para cada dim
            # Solo queremos paddear las últimas 2 dims
            pads = [(0,0)] * (arr.ndim - 2) + [(pad_top, pad_bottom), (pad_left, pad_right)]
            arr = np.pad(arr, pads, mode='constant', constant_values=0)
            
            # Recalcular coords sobre el padded array
            y1 += pad_top
            y2 += pad_top
            x1 += pad_left
            x2 += pad_left
            
        return arr[..., y1:y2, x1:x2]

    def __getitem__(self, idx):
        if self.preload_ram:
            return self.ram_cache[idx]
        return self._load_sample(idx)

    def get_channel_info(self):
        channels = self.feature_vars.copy()
        if self.include_fire_state:
            channels.append("fire_state")
        print("🔍 INFORMACIÓN DE CANALES:")
        for i, ch in enumerate(channels):
            print(f"   Canal {i:2d}: {ch}")
        return channels

    def get_sample_stats(self, n_samples=100):
        """Estadísticas del dataset para verificar calidad"""
        if len(self.indices) == 0:
            print("❌ No hay muestras en el dataset")
            return {}
        
        n_samples = min(n_samples, len(self.indices))
        fire_pixels_current = []
        fire_pixels_next = []
        
        print(f"📊 Analizando {n_samples} muestras...")
        
        for i in range(n_samples):
            x, y = self[i]
            
            if self.include_fire_state:
                # Último canal = estado actual del fuego en último timestep
                current_fire = x[-1, -1, :, :]  # (H, W)
                fire_pixels_current.append(current_fire.sum().item())
            
            fire_pixels_next.append(y.sum().item())
        
        stats = {
            'samples_analyzed': n_samples,
            'avg_fire_pixels_current': np.mean(fire_pixels_current) if fire_pixels_current else 0,
            'avg_fire_pixels_next': np.mean(fire_pixels_next),
            'max_fire_pixels_current': np.max(fire_pixels_current) if fire_pixels_current else 0,
            'max_fire_pixels_next': np.max(fire_pixels_next),
            'samples_with_propagation': sum(1 for fp in fire_pixels_next if fp > 0)
        }
        
        print(f"   🔥 Píxeles de fuego actuales (promedio): {stats['avg_fire_pixels_current']:.1f}")
        print(f"   🔥 Píxeles de fuego t+1 (promedio): {stats['avg_fire_pixels_next']:.1f}")
        print(f"   📈 Muestras con propagación: {stats['samples_with_propagation']}/{n_samples}")
        
        return stats


# Funciones auxiliares para crear y validar datasets
def create_train_val_test_split(datacube, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                               min_temporal_context=3):
    """
    Crea splits temporales para entrenamiento/validación/test
    
    Args:
        datacube: xarray Dataset con dimensión 'time'
        train_ratio, val_ratio, test_ratio: Proporciones de split
        min_temporal_context: Mínimo contexto temporal necesario
    
    Returns:
        dict: {'train': indices, 'val': indices, 'test': indices}
    """
    total_times = len(datacube.time)
    
    # Índices válidos (con suficiente contexto temporal)
    valid_time_indices = list(range(min_temporal_context, total_times - 1))
    n_valid = len(valid_time_indices)
    
    # Splits temporales (no aleatorios para evitar data leakage)
    train_end = int(n_valid * train_ratio)
    val_end = int(n_valid * (train_ratio + val_ratio))
    
    train_times = valid_time_indices[:train_end]
    val_times = valid_time_indices[train_end:val_end]
    test_times = valid_time_indices[val_end:]
    
    def create_indices_list(time_indices):
        return [{"time_index": t} for t in time_indices]
    
    splits = {
        'train': create_indices_list(train_times),
        'val': create_indices_list(val_times),
        'test': create_indices_list(test_times)
    }
    
    print(f"📊 DIVISIÓN TEMPORAL DE DATOS:")
    print(f"   Total timesteps válidos: {n_valid}")
    print(f"   Train: {len(train_times)} timesteps ({len(train_times)/n_valid*100:.1f}%)")
    print(f"   Val:   {len(val_times)} timesteps ({len(val_times)/n_valid*100:.1f}%)")
    print(f"   Test:  {len(test_times)} timesteps ({len(test_times)/n_valid*100:.1f}%)")
    
    return splits


def validate_datasets(ignition_dataset, spread_dataset):
    """
    Valida la compatibilidad entre datasets de ignición y propagación
    """
    print("🔍 VALIDANDO COMPATIBILIDAD DE DATASETS...")
    
    # Verificar shapes
    x_ign, y_ign = ignition_dataset[0]
    x_spr, y_spr = spread_dataset[0]
    
    print(f"   📐 IgnitionDataset:")
    print(f"      Input shape:  {x_ign.shape}")  # (T, C, H, W)
    print(f"      Output shape: {y_ign.shape}")  # (1,)
    
    print(f"   📐 SpreadDataset:")
    print(f"      Input shape:  {x_spr.shape}")  # (T, C+1, H, W)
    print(f"      Output shape: {y_spr.shape}")  # (1, H, W)
    
    # Verificar dimensiones espaciales
    ign_spatial = x_ign.shape[-2:]
    spr_spatial = x_spr.shape[-2:]
    
    if ign_spatial == spr_spatial:
        print(f"   ✅ Dimensiones espaciales compatibles: {ign_spatial}")
    else:
        print(f"   ❌ ERROR: Dimensiones espaciales incompatibles: {ign_spatial} vs {spr_spatial}")
    
    # Verificar número de canales
    ign_channels = x_ign.shape[1]
    spr_channels = x_spr.shape[1]
    
    print(f"   📊 Canales ignición: {ign_channels}")
    print(f"   📊 Canales propagación: {spr_channels} (debe ser {ign_channels}+1)")
    
    if spr_channels == ign_channels + 1:
        print(f"   ✅ Número de canales correcto")
    else:
        print(f"   ⚠️ Advertencia: Número de canales inesperado")
    
    return True


# Ejemplo de uso completo
def setup_datasets_example(datacube):
    """
    Ejemplo completo de cómo configurar los datasets
    """
    print("🚀 CONFIGURANDO DATASETS...")
    
    # 1. Crear splits temporales
    splits = create_train_val_test_split(datacube, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # 2. Crear datasets de ignición
    print("\n🔥 CREANDO DATASETS DE IGNICIÓN...")
    ign_train = IgnitionDataset(datacube, splits['train'], temporal_context=7)
    ign_val = IgnitionDataset(datacube, splits['val'], temporal_context=7)
    ign_test = IgnitionDataset(datacube, splits['test'], temporal_context=7)
    
    # 3. Crear datasets de propagación (CON estado de fuego)
    print("\n🔥 CREANDO DATASETS DE PROPAGACIÓN...")
    spr_train = SpreadDataset(datacube, splits['train'], temporal_context=3, 
                             include_fire_state=True, filter_fire_samples=True)
    spr_val = SpreadDataset(datacube, splits['val'], temporal_context=3, 
                           include_fire_state=True, filter_fire_samples=True)
    spr_test = SpreadDataset(datacube, splits['test'], temporal_context=3, 
                            include_fire_state=True, filter_fire_samples=True)
    
    # 4. Validar compatibilidad
    print("\n🔍 VALIDACIÓN...")
    validate_datasets(ign_train, spr_train)
    
    # 5. Estadísticas
    print("\n📊 ESTADÍSTICAS:")
    spr_train.get_sample_stats(n_samples=100)
    
    return {
        'ignition': {'train': ign_train, 'val': ign_val, 'test': ign_test},
        'spread': {'train': spr_train, 'val': spr_val, 'test': spr_test}
    }


import pandas as pd
import numpy as np

def create_year_split(datacube, train_years, val_years, test_years, min_temporal_context=3):
    """
    Crea splits basados en años explícitos.
    
    Args:
        datacube: xarray Dataset
        train_years: Lista de años para train [2015, ..., 2021]
        val_years: Lista de años para val [2022, 2023]
        test_years: Lista de años para test [2024]
        min_temporal_context: Margen de seguridad temporal
    """
    splits = {'train': [], 'val': [], 'test': []}
    
    print(f"🔄 Generando split por años:")
    print(f"   Train: {train_years}")
    print(f"   Val:   {val_years}")
    print(f"   Test:  {test_years}")

    # Convertir a sets para búsqueda rápida
    train_set = set(train_years)
    val_set = set(val_years)
    test_set = set(test_years)
    
    times = datacube.time.values
    
    # Empezamos desde min_temporal_context para asegurar historial
    for t_idx in range(min_temporal_context, len(times) - 1):
        # Obtener año del timestamp actual
        ts = pd.to_datetime(times[t_idx])
        year = ts.year
        
        sample = {"time_index": t_idx}
        
        if year in train_set:
            splits['train'].append(sample)
        elif year in val_set:
            splits['val'].append(sample)
        elif year in test_set:
            splits['test'].append(sample)
            
    print(f"📊 Muestras generadas:")
    print(f"   Test:  {len(splits['test'])}")
    
    return splits

def create_spatial_split(datacube, val_region='east', split_ratio=0.2):
    """
    Crea máscaras espaciales para validación geográfica (West vs East).
    
    Args:
        datacube: xarray Dataset
        val_region: 'east', 'west', 'north', 'south'
        split_ratio: Proporción del área para validación
        
    Returns:
        dict: {'train': mask_train, 'val': mask_val} (Arrays booleanos 2D)
    """
    height = datacube.sizes['y']
    width = datacube.sizes['x']
    
    # Crear grid de coordenadas
    y_indices = np.arange(height)
    x_indices = np.arange(width)
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Inicializar máscaras
    mask_val = np.zeros((height, width), dtype=bool)
    
    if val_region == 'east':
        # Validar en el Este (x > threshold)
        threshold = int(width * (1 - split_ratio))
        mask_val[:, threshold:] = True
    elif val_region == 'west':
        # Validar en el Oeste (x < threshold)
        threshold = int(width * split_ratio)
        mask_val[:, :threshold] = True
    elif val_region == 'south':
        # Validar en el Sur (y < threshold, asumiendo y crece hacia arriba o abajo verificar coords)
        # Ojo: Indices y suelen ir 0..H. Coordenadas pueden ser latitud.
        # Usamos indices para ser agnósticos.
        threshold = int(height * split_ratio)
        mask_val[threshold:, :] = True # Asumiendo 0 es arriba (norte) en imagen
    elif val_region == 'north':
        threshold = int(height * (1 - split_ratio))
        mask_val[:threshold, :] = True
        
    mask_train = ~mask_val
    
    print(f"🌍 Generando split espacial ({val_region}):")
    print(f"   Train pixels: {np.sum(mask_train)} ({np.sum(mask_train)/(width*height)*100:.1f}%)")
    print(f"   Val pixels:   {np.sum(mask_val)} ({np.sum(mask_val)/(width*height)*100:.1f}%)")
    
    return {
        'train': mask_train,
        'val': mask_val
    }

def generate_temporal_splits(datacube, strict=True):
    """
    Helper para generar splits temporales estándar del proyecto.
    Train: 2009-2019
    Val: 2020-2021
    Test: 2022
    """
    if strict:
        # Recuperar años disponibles
        years = sorted(list(set(pd.to_datetime(datacube.time.values).year)))
        
        train_years = [y for y in years if y <= 2020]
        val_years = [y for y in years if 2021 <= y <= 2022]
        # Include 2023, 2024 in test
        test_years = [y for y in years if y >= 2023]
        
        return create_year_split(datacube, train_years, val_years, test_years)
    else:
        # Fallback o random (no implementado aquí por ahora)
        raise NotImplementedError("Solo strict=True está implementado para consistencia.")

def generate_spatial_splits(datacube, test_region="east"):
    """
    Helper para splits espaciales (wrapper de create_spatial_split).
    """
    masks = create_spatial_split(datacube, val_region=test_region)
    
    # SOLUCIÓN: Retornar todos los tiempos, y que el Dataset aplique la máscara espacial
    all_indices = [{'time_index': t} for t in range(len(datacube.time))]
    return {
        'train': all_indices, 
        'val': all_indices, 
        'masks': masks # Pasamos las máscaras aparte
    }
