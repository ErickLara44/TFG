#!/usr/bin/env python3
"""
Computes mean/std for every variable in DEFAULT_FEATURE_VARS (spread dataset)
from the datacube and saves to JSON so training scripts can import them.

Usage:
    python scripts/compute_spread_stats.py
    python scripts/compute_spread_stats.py --samples 1000
    python scripts/compute_spread_stats.py --datacube other.nc --output tmp/stats.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_prop_improved import DEFAULT_FEATURE_VARS, DEFAULT_STATS_PATH

DEFAULT_N_TIME_SAMPLES = 200   # suficiente para mean/std estables con spatial stride
DEFAULT_SPATIAL_STRIDE = 4     # 1 de cada 16 píxeles (920x1188 → 230x297 ≈ 68k puntos/frame)


def _time_slice(n_time, n_samples):
    """Stride contiguo sobre el eje temporal (evita reads aleatorios lentos en NetCDF)."""
    stride = max(1, n_time // max(1, n_samples))
    return slice(0, n_time, stride)


def _read_temporal(ds, name, t_slice, sp):
    return ds[name].isel(time=t_slice,
                         y=slice(None, None, sp),
                         x=slice(None, None, sp)).values


def _read_static(ds, name, sp):
    return ds[name].isel(y=slice(None, None, sp), x=slice(None, None, sp)).values


def _compute_virtual(var, ds, t_slice, sp):
    """Reproduce la lógica de SpreadDataset._read_variable_smart para estadísticas globales."""
    if var in ('wind_u', 'wind_v'):
        ws = _read_temporal(ds, 'wind_speed_mean', t_slice, sp)
        wd = _read_temporal(ds, 'wind_direction_mean', t_slice, sp)
        wd_rad = np.deg2rad(wd)
        return ws * np.sin(wd_rad) if var == 'wind_u' else ws * np.cos(wd_rad)

    if var == 'hydric_stress':
        t2m = _read_temporal(ds, 't2m_mean', t_slice, sp)
        lst = _read_temporal(ds, 'LST', t_slice, sp)
        if np.nanmean(lst) > 200 and np.nanmean(t2m) < 100:
            return lst - (t2m + 273.15)
        return lst - t2m

    if var == 'solar_risk':
        t2m_max = _read_temporal(ds, 't2m_max', t_slice, sp)
        south = np.zeros_like(t2m_max)
        for a in ('aspect_4', 'aspect_5', 'aspect_6'):
            if a in ds.data_vars:
                south = south + _read_static(ds, a, sp)[None, ...]
        return t2m_max * south

    if var.startswith('CLC_current_'):
        # Mezcla temporal: cada año apunta a la edición de CLC correspondiente.
        suffix = var[len('CLC_current_'):]
        years = pd.to_datetime(ds.time.isel(time=t_slice).values).year.to_numpy()
        buckets = {'CLC_2006': years <= 2009,
                   'CLC_2012': (years > 2009) & (years <= 2015),
                   'CLC_2018': years > 2015}
        parts = []
        for prefix, mask in buckets.items():
            if not mask.any():
                continue
            name = f"{prefix}_{suffix}"
            if name in ds.data_vars:
                vals = _read_static(ds, name, sp).flatten()
                parts.append(np.repeat(vals[None, :], int(mask.sum()), axis=0))
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    return None


def compute_stats(ds, feature_vars,
                  n_time_samples=DEFAULT_N_TIME_SAMPLES,
                  spatial_stride=DEFAULT_SPATIAL_STRIDE):
    n_time = ds.sizes['time']
    t_slice = _time_slice(n_time, n_time_samples)
    sp = max(1, int(spatial_stride))

    n_time_used = len(range(*t_slice.indices(n_time)))
    n_y = len(range(0, ds.sizes['y'], sp))
    n_x = len(range(0, ds.sizes['x'], sp))
    print(f"   stride temporal={t_slice.step} → {n_time_used} timesteps; "
          f"stride espacial={sp} → {n_y}×{n_x} px por frame\n")

    stats = {}
    for var in feature_vars:
        print(f"  {var:40s} ", end='', flush=True)

        if var in ds.data_vars:
            da = ds[var]
            vals = _read_temporal(ds, var, t_slice, sp) if 'time' in da.dims else _read_static(ds, var, sp)
        else:
            vals = _compute_virtual(var, ds, t_slice, sp)
            if vals is None:
                print("⚠️  no disponible, skip")
                continue

        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        if not np.isfinite(mean) or not np.isfinite(std):
            print(f"⚠️  valores no finitos (mean={mean}, std={std}), skip")
            continue
        stats[var] = {'mean': mean, 'std': std}
        print(f"mean={mean: .4f}  std={std: .4f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datacube', default='data/IberFire.nc')
    parser.add_argument('--output', default=str(DEFAULT_STATS_PATH))
    parser.add_argument('--samples', type=int, default=DEFAULT_N_TIME_SAMPLES,
                        help=f'Número de timesteps a muestrear (default {DEFAULT_N_TIME_SAMPLES})')
    parser.add_argument('--spatial-stride', type=int, default=DEFAULT_SPATIAL_STRIDE,
                        help=f'Stride espacial para subsample (default {DEFAULT_SPATIAL_STRIDE})')
    args = parser.parse_args()

    print(f"📂 Abriendo {args.datacube}")
    ds = xr.open_dataset(args.datacube)

    print(f"⚙️  Computando stats para {len(DEFAULT_FEATURE_VARS)} vars "
          f"({args.samples} timesteps, stride espacial {args.spatial_stride})")
    stats = compute_stats(ds, DEFAULT_FEATURE_VARS,
                          n_time_samples=args.samples,
                          spatial_stride=args.spatial_stride)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Guardado en {out_path} ({len(stats)}/{len(DEFAULT_FEATURE_VARS)} vars cubiertas)")


if __name__ == '__main__':
    main()
