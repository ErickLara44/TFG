#!/usr/bin/env python3
"""
Computes mean/std for every variable in DEFAULT_FEATURE_VARS (ignition dataset)
from the datacube and saves to JSON so training/eval/viz scripts can import them.

Usage:
    python scripts/compute_ignition_stats.py
    python scripts/compute_ignition_stats.py --samples 500
    python scripts/compute_ignition_stats.py --datacube other.nc --output tmp/stats.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS, DEFAULT_STATS_PATH, VIRTUAL_TIME_VARS

DEFAULT_N_TIME_SAMPLES = 200
DEFAULT_SPATIAL_STRIDE = 4


def _time_slice(n_time, n_samples):
    stride = max(1, n_time // max(1, n_samples))
    return slice(0, n_time, stride)


def _read_temporal(ds, name, t_slice, sp):
    return ds[name].isel(time=t_slice,
                         y=slice(None, None, sp),
                         x=slice(None, None, sp)).values


def _read_static(ds, name, sp):
    return ds[name].isel(y=slice(None, None, sp), x=slice(None, None, sp)).values


def _compute_virtual(var, ds, t_slice):
    """Variables de calendario (1D sobre time). Constantes espacialmente."""
    if var not in VIRTUAL_TIME_VARS:
        return None
    times = pd.to_datetime(ds.time.isel(time=t_slice).values)
    doy = times.dayofyear.values.astype(np.float32)
    two_pi = 2.0 * np.pi
    if var == 'is_weekend':
        return (times.dayofweek.values >= 5).astype(np.float32)
    if var == 'day_of_year_sin':
        return np.sin(two_pi * doy / 365.25).astype(np.float32)
    if var == 'day_of_year_cos':
        return np.cos(two_pi * doy / 365.25).astype(np.float32)
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

        if var in VIRTUAL_TIME_VARS:
            vals = _compute_virtual(var, ds, t_slice)
        elif var in ds.data_vars:
            da = ds[var]
            vals = _read_temporal(ds, var, t_slice, sp) if 'time' in da.dims else _read_static(ds, var, sp)
        else:
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
