import xarray as xr
import numpy as np
import pandas as pd
from loguru import logger
import gc

def compute_derived_features(ds: xr.Dataset) -> xr.Dataset:
    """
    Enriquece el datacube con variables físicas avanzadas y correcciones de metodología.
    Versión optimizada para memoria.
    
    Args:
        ds: xarray.Dataset original
        
    Returns:
        xarray.Dataset enriquecido
    """
    logger.info("🛠️  Iniciando ingeniería de características avanzada (Física de Incendios)...")
    
    # 1. Dynamic Land Cover (Selección Temporal) 🌿
    try:
        logger.info("   🌿 Generando cobertura terrestre dinámica (CLC)...")
        clc_types = ['forest_proportion', 'scrub_proportion', 'agricultural_proportion', 
                     'urban_fabric_proportion', 'waterbody_proportion']

        years = ds['time.year']
        
        # Pre-calcular máscaras temporales
        mask_2006 = (years <= 2009).values
        mask_2012 = ((years > 2009) & (years <= 2015)).values
        
        for c_type in clc_types:
            var_2006 = f"CLC_2006_{c_type}"
            var_2012 = f"CLC_2012_{c_type}"
            var_2018 = f"CLC_2018_{c_type}"
            
            if not all(v in ds for v in [var_2006, var_2012, var_2018]):
                continue
                
            # Crear array vacío (time, y, x)
            # ds[var_2018] es (y, x) estático. Necesitamos expandirlo a (time, y, x)
            t_dim = ds.sizes['time']
            y_dim = ds.sizes['y']
            x_dim = ds.sizes['x']
            
            new_var_data = np.zeros((t_dim, y_dim, x_dim), dtype=np.float32)
            
            # Asignar por slices temporales (Broadcasting de (y,x) a (N_subset, y, x))
            if np.any(mask_2006):
                new_var_data[mask_2006] = ds[var_2006].values
            
            if np.any(mask_2012):
                new_var_data[mask_2012] = ds[var_2012].values
                
            mask_2018 = ~mask_2006 & ~mask_2012
            if np.any(mask_2018):
                new_var_data[mask_2018] = ds[var_2018].values
            
            ds[f"CLC_current_{c_type}"] = (('time', 'y', 'x'), new_var_data)
            
            # Liberar memoria explícitamente
            del new_var_data
            gc.collect()

        logger.success("   ✅ Dynamic CLC completado.")

        # 2. Hydric Stress (LST vs Air Temp) 🌡️
        if 'LST' in ds and 't2m_mean' in ds:
            logger.info("   🔥 Calculando Estrés Hídrico...")
            t2m = ds['t2m_mean'].values
            lst = ds['LST'].values
            
            # Heurística simple K vs C
            if lst.mean() > 200 and t2m.mean() < 100:
                stress = lst - (t2m + 273.15)
            else:
                stress = lst - t2m
                
            ds['hydric_stress'] = (('time', 'y', 'x'), stress.astype(np.float32))
            del stress, t2m, lst
            gc.collect()

        # 3. Solar Risk & Wind
        logger.info("   ☀️💨 Calculando Riesgo Solar y Viento...")
        
        # Solar Risk
        if 't2m_max' in ds:
            south_aspect = np.zeros_like(ds['t2m_max'].isel(time=0).values, dtype=np.float32)
            for a in ['aspect_4', 'aspect_5', 'aspect_6']:
                if a in ds:
                    south_aspect += ds[a].values
            
            ds['solar_risk'] = ds['t2m_max'] * south_aspect
            del south_aspect
            gc.collect()

        # Wind Vectors
        if 'wind_speed_mean' in ds and 'wind_direction_mean' in ds:
            wd_rad = np.deg2rad(ds['wind_direction_mean'].values)
            ws = ds['wind_speed_mean'].values
            
            ds['wind_u'] = (('time', 'y', 'x'), (ws * np.sin(wd_rad)).astype(np.float32))
            ds['wind_v'] = (('time', 'y', 'x'), (ws * np.cos(wd_rad)).astype(np.float32))
            del wd_rad, ws
            gc.collect()

        # 4. Structural Drought & Anti-Leakage
        if 'SWI_020' in ds and 'SWI_001' in ds:
             # Calcular chunk a chunk o todo si cabe (SWI suele ser ligero)
            ds['structural_drought'] = (ds['SWI_020'] - ds['SWI_001']).astype(np.float32)
            
        if 'is_near_fire' in ds:
            logger.info("   � Aplicando lag a is_near_fire...")
            ds['is_near_fire_lag1'] = ds['is_near_fire'].shift(time=1).fillna(0).astype(np.float32)

    except Exception as e:
        logger.error(f"❌ Error en ingeniería de características: {e}")
        return ds

    logger.success("✅ Datacube enriquecido (Optimized).")
    return ds
