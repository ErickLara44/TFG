import xarray as xr
import pyproj
import numpy as np
import datetime
from pathlib import Path
import os
import sys
import pickle
import httpx

# Add project root to path if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DATACUBE_PATH
from src.api.fwi import compute_fwi

# Globals
__datacube = None
__transformer_to_3035 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
__transformer_to_4326 = pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
__scaler = None

SCALER_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "scaler.pkl"
IGNITION_STATS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "iberfire_normalization_stats.pkl"

# --- DEFINICIONES DE CANALES (Mapping Oficial del Entrenamiento) ---

IGNITION_CHANNELS = [
    'elevation_mean', 'slope_mean',
    'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion', 'CLC_2018_agricultural_proportion',
    'dist_to_roads_mean', 'popdens_2018', 'is_waterbody',
    't2m_mean', 'RH_min', 'wind_speed_mean', 'total_precipitation_mean',
    'NDVI', 'SWI_010', 'FWI', 'LST', 'wind_direction_mean'
]

SPREAD_CHANNELS = [
    'elevation_mean', 'slope_mean',
    'wind_u', 'wind_v',
    'hydric_stress',
    'solar_risk',
    'CLC_current_forest_proportion', 
    'CLC_current_scrub_proportion',
    'FWI', 'NDVI',
    'dist_to_roads_mean',
    'fire_state'
]

# Estadísticas para el modelo de Propagación (Training Script Stats)
SPREAD_STATS = {
    'elevation_mean': {'mean': 554.24, 'std': 418.34},
    'slope_mean': {'mean': 8.31, 'std': 6.72},
    'wind_u': {'mean': 0.0, 'std': 5.0}, 
    'wind_v': {'mean': 0.0, 'std': 5.0},
    'hydric_stress': {'mean': 0.0, 'std': 10.0},
    'solar_risk': {'mean': 15.0, 'std': 10.0},
    'CLC_current_forest_proportion': {'mean': 0.24, 'std': 0.32},
    'CLC_current_scrub_proportion': {'mean': 0.23, 'std': 0.31},
    'FWI': {'mean': 14.09, 'std': 16.41},
    'NDVI': {'mean': 0.47, 'std': 0.18},
    'dist_to_roads_mean': {'mean': 1.22, 'std': 1.28},
}

__ignition_stats = None

def get_ignition_stats():
    global __ignition_stats
    if __ignition_stats is None:
        if IGNITION_STATS_PATH.exists():
            with open(IGNITION_STATS_PATH, "rb") as f:
                __ignition_stats = pickle.load(f)
        else:
            print(f"⚠️ Ignition stats not found at {IGNITION_STATS_PATH}")
    return __ignition_stats

def get_scaler():
    global __scaler
    if __scaler is None:
        if SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                __scaler = pickle.load(f)
        else:
            print(f"⚠️ Scaler not found at {SCALER_PATH}")
    return __scaler

def get_datacube():
    global __datacube
    if __datacube is None:
        try:
            __datacube = xr.open_dataset(DATACUBE_PATH)
        except Exception as e:
            print(f"Error loading Datacube: {e}")
    return __datacube

# --- CLIMATE FETCHING ---

async def fetch_climate_window(lat: float, lon: float, end_date: str, n_days: int = 7) -> list[dict]:
    target = datetime.date.fromisoformat(end_date)
    start = target - datetime.timedelta(days=n_days - 1)
    today = datetime.date.today()
    url = "https://archive-api.open-meteo.com/v1/archive" if target <= today else "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": target.isoformat(),
        "daily": ["temperature_2m_mean", "temperature_2m_max", "relative_humidity_2m_mean", "precipitation_sum"],
        "hourly": ["wind_speed_10m", "wind_direction_10m"], "wind_speed_unit": "ms", "timezone": "Europe/Madrid"
    }
    fallback = [{"t2m_mean": 20.0, "t2m_max": 25.0, "RH_min": 50.0, "wind_speed_mean": 5.0, "wind_direction_mean": 0.0, "total_precipitation_mean": 0.0} for _ in range(n_days)]
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        daily = data.get("daily", {}); hourly = data.get("hourly", {})
        dates = daily.get("time", []); t_mean = daily.get("temperature_2m_mean", []); t_max = daily.get("temperature_2m_max", [])
        rh = daily.get("relative_humidity_2m_mean", []); precip = daily.get("precipitation_sum", [])
        h_spd = hourly.get("wind_speed_10m", []); h_dir = hourly.get("wind_direction_10m", [])
        records = []
        for i, d in enumerate(dates):
            spd_slice = [v for v in h_spd[i*24:(i+1)*24] if v is not None]
            dir_slice = [v for v in h_dir[i*24:(i+1)*24] if v is not None]
            records.append({
                "date": d, "t2m_mean": t_mean[i] or 20.0, "t2m_max": t_max[i] or 25.0,
                "RH_min": rh[i] or 50.0, "wind_speed_mean": float(np.mean(spd_slice)) if spd_slice else 5.0,
                "wind_direction_mean": float(np.mean(dir_slice)) if dir_slice else 0.0, "total_precipitation_mean": precip[i] or 0.0
            })
        return records
    except Exception: return fallback

def compute_fwi_window(climate_window: list[dict]) -> float:
    p_ffmc, p_dmc, p_dc = 85.0, 6.0, 15.0
    fwi_val = 0.0
    for day in climate_window:
        d = datetime.date.fromisoformat(day["date"])
        res = compute_fwi(temp=day["t2m_mean"], rh=day["RH_min"], wind_kmh=day["wind_speed_mean"]*3.6, rain=day["total_precipitation_mean"],
                            month=d.month, prev_ffmc=p_ffmc, prev_dmc=p_dmc, prev_dc=p_dc)
        p_ffmc, p_dmc, p_dc = res["FFMC"], res["DMC"], res["DC"]
        fwi_val = res["FWI"]
    return fwi_val

# --- TENSOR BUILDING ENGINES ---

async def get_ignition_tensor(lat: float, lon: float, date: str) -> tuple[np.ndarray, dict]:
    ds = get_datacube(); x_3035, y_3035 = __transformer_to_3035.transform(lon, lat)
    point_data = ds.sel(x=x_3035, y=y_3035, method="nearest")
    cx = np.abs(ds.x.values - x_3035).argmin(); cy = np.abs(ds.y.values - y_3035).argmin()
    
    half = 32; y_s = max(0, cy-half); y_e = min(ds.sizes['y'], cy+half); x_s = max(0, cx-half); x_e = min(ds.sizes['x'], cx+half)
    
    # --- TEMPORAL MAPPING FIX ---
    target_dt = np.datetime64(date)
    max_dc_dt = ds.time.values[-1]
    
    if target_dt > max_dc_dt:
        # If requested date is beyond Datacube (e.g. 2025), map to same day/month in last available year
        req_d = datetime.date.fromisoformat(date)
        last_y_d = datetime.date(2021, req_d.month, req_d.day)
        t_idx = np.abs(ds.time.values - np.datetime64(last_y_d)).argmin()
    else:
        t_idx = np.abs(ds.time.values - target_dt).argmin()
    # ----------------------------
    
    climate_window = await fetch_climate_window(lat, lon, date, 7)
    today = climate_window[-1]
    fwi_val = compute_fwi_window(climate_window)
    
    T, C, H, W = 3, 18, y_e - y_s, x_e - x_s
    raw = np.zeros((T, C, H, W), dtype=np.float32)
    last_3_clim = climate_window[-3:]
    
    for c, var in enumerate(IGNITION_CHANNELS):
        # 1. Variables estáticas del Datacube
        if var in ['elevation_mean', 'slope_mean', 'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion', 'CLC_2018_agricultural_proportion', 'dist_to_roads_mean', 'popdens_2018', 'is_waterbody']:
            if var in ds.data_vars:
                spatial = ds[var].isel(y=slice(y_s, y_e), x=slice(x_s, x_e)).values
                if '_proportion' in var and np.nanmax(spatial) > 1.1: spatial = spatial / 100.0
                for t in range(T): raw[t, c, :, :] = spatial
        # 2. Variables dinámicas del Datacube
        elif var in ['NDVI', 'SWI_010', 'LST']:
            if var in ds.data_vars:
                if 'time' in ds[var].dims:
                    t_s_dc = max(0, t_idx - 2); t_e_dc = t_idx + 1
                    slice_dc = ds[var].isel(time=slice(t_s_dc, t_e_dc), y=slice(y_s, y_e), x=slice(x_s, x_e)).values
                    for t in range(min(T, slice_dc.shape[0])): raw[t, c, :, :] = slice_dc[t]
                    # Fix frozen LST if needed
                    if var == 'LST':
                        for t in range(T):
                            if np.nanmean(raw[t, c, :, :]) < 270: 
                                raw[t, c, :, :] = (last_3_clim[t].get('t2m_mean', 20) + 273.15) + 5.0
                else: 
                    spatial = ds[var].isel(y=slice(y_s, y_e), x=slice(x_s, x_e)).values
                    for t in range(T): raw[t, c, :, :] = spatial
        # 3. Climatología en tiempo real
        elif var in ['t2m_mean', 'RH_min', 'wind_speed_mean', 'total_precipitation_mean', 'wind_direction_mean']:
            for t in range(T): raw[t, c, :, :] = last_3_clim[t].get(var, 0.0)
        # 4. FWI
        elif var == 'FWI':
            for t in range(T): raw[t, c, :, :] = fwi_val
    # Padding a 64x64
    if H < 64 or W < 64:
        padded = np.zeros((3, 18, 64, 64), dtype=np.float32)
        dy, dx = (64 - H) // 2, (64 - W) // 2
        padded[:, :, dy:dy+H, dx:dx+W] = raw
        raw = padded
        
    # Normalización con stats reales del entrenamiento (iberfire_normalization_stats.pkl)
    ign_stats = get_ignition_stats()
    if ign_stats:
        m_vec = np.array(ign_stats['mean'], dtype=np.float32)  # (18,)
        s_vec = np.array(ign_stats['std'], dtype=np.float32)   # (18,)
        raw = (raw - m_vec.reshape(1, 18, 1, 1)) / (s_vec.reshape(1, 18, 1, 1) + 1e-8)
        
    # --- METADATA FOR FRONTEND ---
    # Extract real values from the center pixel for UI display
    ext_feat = {
        "FWI": fwi_val, 
        "humidity": today["RH_min"], 
        "temperature": today["t2m_mean"],
        "wind_speed": today["wind_speed_mean"],
        "wind_direction": today["wind_direction_mean"],
        "elevation": float(ds['elevation_mean'].sel(x=x_3035, y=y_3035, method="nearest").values),
        "slope": float(ds['slope_mean'].sel(x=x_3035, y=y_3035, method="nearest").values),
        "forest_prop": float(ds['CLC_2018_forest_proportion'].sel(x=x_3035, y=y_3035, method="nearest").values),
        "scrub_prop": float(ds['CLC_2018_scrub_proportion'].sel(x=x_3035, y=y_3035, method="nearest").values),
    }
    
    # Calculate cell bounds (1km patch approx)
    res = 500 # 500m from center
    p1 = __transformer_to_4326.transform(x_3035 - res, y_3035 - res)
    p2 = __transformer_to_4326.transform(x_3035 - res, y_3035 + res)
    p3 = __transformer_to_4326.transform(x_3035 + res, y_3035 + res)
    p4 = __transformer_to_4326.transform(x_3035 + res, y_3035 - res)
    cell_bounds = [[p1[1], p1[0]], [p2[1], p2[0]], [p3[1], p3[0]], [p4[1], p4[0]], [p1[1], p1[0]]]

    return np.nan_to_num(raw[np.newaxis, ...], nan=0.0), {
        "features": ext_feat,
        "x_3035": float(x_3035),
        "y_3035": float(y_3035),
        "cell_bounds": cell_bounds
    }

async def get_spread_tensor(lat: float, lon: float, date: str) -> tuple[np.ndarray, dict]:
    ds = get_datacube(); x_3035, y_3035 = __transformer_to_3035.transform(lon, lat)
    cx = np.abs(ds.x.values - x_3035).argmin(); cy = np.abs(ds.y.values - y_3035).argmin()
    half = 32; y_s = max(0, cy-half); y_e = min(ds.sizes['y'], cy+half); x_s = max(0, cx-half); x_e = min(ds.sizes['x'], cx+half)
    
    climate_window = await fetch_climate_window(lat, lon, date, 7)
    today = climate_window[-1]
    fwi_val = compute_fwi_window(climate_window)
    last_3_clim = climate_window[-3:]
    
    T, C, H, W = 3, 12, y_e - y_s, x_e - x_s
    raw = np.zeros((T, C, H, W), dtype=np.float32)
    
    # Pre-cargar LST para Hydric Stress
    target_dt = np.datetime64(date); t_idx = np.abs(ds.time.values - target_dt).argmin()
    lst_map = ds['LST'].isel(time=t_idx, y=slice(y_s, y_e), x=slice(x_s, x_e)).values if 'LST' in ds.data_vars else np.zeros((H,W))
    if np.nanmean(lst_map) < 270: lst_map = (today['t2m_mean'] + 273.15) + 5.0
    
    for c, var in enumerate(SPREAD_CHANNELS):
        if var in ['elevation_mean', 'slope_mean', 'dist_to_roads_mean']:
            if var in ds.data_vars: raw[:, c, :H, :W] = ds[var].isel(y=slice(y_s, y_e), x=slice(x_s, x_e)).values
        elif var == 'wind_u':
            for t in range(T):
                ws = last_3_clim[t]['wind_speed_mean']; wd = last_3_clim[t]['wind_direction_mean']
                raw[t, c, :, :] = ws * np.sin(np.deg2rad(wd))
        elif var == 'wind_v':
            for t in range(T):
                ws = last_3_clim[t]['wind_speed_mean']; wd = last_3_clim[t]['wind_direction_mean']
                raw[t, c, :, :] = ws * np.cos(np.deg2rad(wd))
        elif var == 'hydric_stress':
            for t in range(T): raw[t, c, :, :] = np.clip(lst_map - (last_3_clim[t]['t2m_mean'] + 273.15), -20, 20)
        elif var == 'solar_risk':
            for t in range(T): raw[t, c, :, :] = last_3_clim[t]['t2m_max'] * 0.3
        elif var == 'CLC_current_forest_proportion':
            raw[:, c, :H, :W] = ds['CLC_2018_forest_proportion'].isel(y=slice(y_s, y_e), x=slice(x_s, x_e)).values / 100.0
        elif var == 'CLC_current_scrub_proportion':
            raw[:, c, :H, :W] = ds['CLC_2018_scrub_proportion'].isel(y=slice(y_s, y_e), x=slice(x_s, x_e)).values / 100.0
        elif var == 'FWI':
            raw[:, c, :, :] = fwi_val
        elif var == 'NDVI':
            raw[:, c, :H, :W] = ds['NDVI'].isel(time=t_idx, y=slice(y_s, y_e), x=slice(x_s, x_e)).values if 'time' in ds['NDVI'].dims else ds['NDVI'].isel(y=slice(y_s, y_e), x=slice(x_s, x_e)).values
        elif var == 'fire_state':
            # 5x5 center fire signal
            cy_c, cx_c = H // 2, W // 2
            raw[:, c, max(0, cy_c-2):min(H, cy_c+3), max(0, cx_c-2):min(W, cx_c+3)] = 1.0

    # Padding a 64x64
    if H < 64 or W < 64:
        padded = np.zeros((3, 12, 64, 64), dtype=np.float32)
        dy, dx = (64 - H) // 2, (64 - W) // 2
        padded[:, :, dy:dy+H, dx:dx+W] = raw
        raw = padded

    # Normalización Específica Spread (Hardcoded Stats from training)
    m_v = np.zeros(12, dtype=np.float32); s_v = np.ones(12, dtype=np.float32)
    for c, var in enumerate(SPREAD_CHANNELS):
        if var in SPREAD_STATS: m_v[c] = SPREAD_STATS[var]['mean']; s_v[c] = SPREAD_STATS[var]['std']
    raw = (raw - m_v.reshape(1, 12, 1, 1)) / (s_v.reshape(1, 12, 1, 1) + 1e-8)
    
    # --- METADATA FOR FRONTEND ---
    ext_feat = {
        "FWI": fwi_val, 
        "t2m": today["t2m_mean"],
        "wind_speed": today["wind_speed_mean"],
        "wind_direction": today["wind_direction_mean"],
        "elevation": float(ds['elevation_mean'].sel(x=x_3035, y=y_3035, method="nearest").values),
        "slope": float(ds['slope_mean'].sel(x=x_3035, y=y_3035, method="nearest").values),
        "forest_prop": float(ds['CLC_2018_forest_proportion'].sel(x=x_3035, y=y_3035, method="nearest").values),
        "scrub_prop": float(ds['CLC_2018_scrub_proportion'].sel(x=x_3035, y=y_3035, method="nearest").values),
    }

    # Calculate cell bounds (64km patch para mapear 64x64 a 1km/pixel)
    res = 32000
    p1 = __transformer_to_4326.transform(x_3035 - res, y_3035 - res)
    p2 = __transformer_to_4326.transform(x_3035 - res, y_3035 + res)
    p3 = __transformer_to_4326.transform(x_3035 + res, y_3035 + res)
    p4 = __transformer_to_4326.transform(x_3035 + res, y_3035 - res)
    cell_bounds = [[p1[1], p1[0]], [p2[1], p2[0]], [p3[1], p3[0]], [p4[1], p4[0]], [p1[1], p1[0]]]

    return np.nan_to_num(raw[np.newaxis, ...], nan=0.0), {
        "features": ext_feat,
        "x_3035": float(x_3035),
        "y_3035": float(y_3035),
        "cell_bounds": cell_bounds
    }

# Main exports remain but point to specialized engines
async def get_features_for_point(lat: float, lon: float, date: str) -> dict:
    """
    Utility function used by main.py to fetch raw/normalized features 
    independent of the tensor building logic.
    """
    _, features = await get_ignition_tensor(lat, lon, date)
    return features

async def get_tensor_for_point(lat: float, lon: float, date: str) -> tuple[np.ndarray, dict]:
    return await get_ignition_tensor(lat, lon, date)

async def get_spread_tensor_for_point(lat: float, lon: float, date: str) -> tuple[np.ndarray, dict]:
    return await get_spread_tensor(lat, lon, date)
