"""
SpainCubeFireDataset - Pipeline tabular para prediccion de incendios.

Extrae datos del datacube NetCDF como muestras puntuales (t, y, x) para
entrenar Random Forest / XGBoost.  Alineado con el pipeline ConvLSTM:
  - Mismas 42 features (DEFAULT_FEATURE_VARS)
  - Mismo split temporal (train <= 2020, val 2021-22, test 2023-24)
  - Mismo ratio neg/pos (2:1 por split)
  - Offset temporal: features en t, label is_fire en t+1 (prediccion next-day)
  - Rechazo de negativos contra coordenadas de fuego

Estrategia rapida (point-based):
  1. Escanear solo 'is_fire' para encontrar eventos (variable ligera)
  2. Muestrear negativos con rechazo, ratio controlado por split
  3. Extraer features SOLO para los puntos seleccionados (RAM minima)
  4. Normalizar (scaler fit en train)
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import gc
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS, VIRTUAL_TIME_VARS  # type: ignore


class SpainCubeFireDataset:
    """
    Pipeline tabular point-based para prediccion de incendios.

    Uso:
        ds = SpainCubeFireDataset("data/IberFire.nc")
        ds.build_dataset()
        ds.normalize_data()
        X_train, y_train, feat_names = ds.get_features_labels("train")
    """

    CCAA_NAMES = {
        1: "Andalucia", 3: "Asturias", 7: "Castilla y Leon",
        10: "Comunidad Valenciana", 11: "Extremadura", 12: "Galicia",
    }

    def __init__(self, datacube_path,
                 output_dir="data/processed/tabular",
                 start_year=2009, end_year=2024,
                 ccaa_target=None,
                 neg_pos_ratio=2.0,
                 train_year_max=2020,
                 val_years=(2021, 2022),
                 test_years=(2023, 2024)):
        """
        Args:
            datacube_path: ruta al NetCDF (IberFire.nc).
            output_dir: directorio de salida para parquets y scaler.
            start_year / end_year: rango temporal a procesar.
            ccaa_target: codigos CCAA a incluir (None = default 6 CCAA).
            neg_pos_ratio: ratio neg/pos POR SPLIT (2.0 = 2 no-fuegos por fuego).
            train_year_max: ultimo anio incluido en train.
            val_years / test_years: tuplas de anios.
        """
        if ccaa_target is None:
            ccaa_target = [12, 3, 7, 11, 10, 1]

        self.datacube_path = datacube_path
        self.output_dir = output_dir
        self.start_year = start_year
        self.end_year = end_year
        self.ccaa_target = ccaa_target
        self.neg_pos_ratio = neg_pos_ratio
        self.train_year_max = train_year_max
        self.val_years = set(val_years)
        self.test_years = set(test_years)

        os.makedirs(output_dir, exist_ok=True)

        self.train_file = os.path.join(output_dir, "train.parquet")
        self.val_file = os.path.join(output_dir, "val.parquet")
        self.test_file = os.path.join(output_dir, "test.parquet")
        self.scaler_path = os.path.join(output_dir, "scaler.pkl")
        self.metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        self.params_path = os.path.join(output_dir, "_params.json")

        self.feature_vars = list(DEFAULT_FEATURE_VARS)

        print("\n" + "=" * 60)
        print("DATASET TABULAR - PREDICCION DE INCENDIOS")
        print("=" * 60)
        print(f"Datacube: {datacube_path}")
        print(f"Periodo: {start_year}-{end_year}")
        print(f"CCAA: {[self.CCAA_NAMES.get(c, c) for c in self.ccaa_target]}")
        print(f"Split: train <={train_year_max} | val {sorted(self.val_years)} | test {sorted(self.test_years)}")
        print(f"Ratio neg/pos: {neg_pos_ratio} (por split)")
        print(f"Features: {len(self.feature_vars)} vars (alineadas con ConvLSTM)")
        print(f"Offset temporal: features(t) -> label(t+1)")
        print("=" * 60 + "\n")

    # ──────────────────────────────────────────────────────────────
    # CACHE INVALIDATION
    # ──────────────────────────────────────────────────────────────

    def _current_params(self):
        """Parametros clave que afectan la salida."""
        return {
            'start_year': self.start_year,
            'end_year': self.end_year,
            'ccaa_target': sorted(self.ccaa_target),
            'neg_pos_ratio': self.neg_pos_ratio,
            'train_year_max': self.train_year_max,
            'val_years': sorted(self.val_years),
            'test_years': sorted(self.test_years),
            'n_features': len(self.feature_vars),
        }

    def _check_cache_valid(self):
        """Verifica si los parquets existentes corresponden a los params actuales."""
        files_exist = all(
            os.path.exists(f) for f in [self.train_file, self.val_file, self.test_file]
        )
        if not files_exist:
            return False
        if not os.path.exists(self.params_path):
            return False
        with open(self.params_path) as f:
            saved = json.load(f)
        return saved == self._current_params()

    def _save_params(self):
        """Guarda params actuales para validacion de cache futura."""
        with open(self.params_path, 'w') as f:
            json.dump(self._current_params(), f, indent=2)

    # ──────────────────────────────────────────────────────────────
    # SPATIAL MASK
    # ──────────────────────────────────────────────────────────────

    def _build_ccaa_mask(self, ds):
        """Mascara booleana 2D (y, x) de celdas en las CCAA objetivo."""
        ccaa = ds["AutonomousCommunities"]
        if "time" in ccaa.dims:
            ccaa = ccaa.isel(time=0)
        mask = np.isin(ccaa.values, self.ccaa_target)
        print(f"   Celdas en CCAA: {mask.sum():,} / {mask.size:,} "
              f"({mask.sum() / mask.size * 100:.1f}%)")
        return mask

    # ──────────────────────────────────────────────────────────────
    # SCAN FIRES (with temporal offset t -> t+1)
    # ──────────────────────────────────────────────────────────────

    def _assign_split(self, year_label):
        """Asigna un anio a su split correspondiente."""
        if year_label <= self.train_year_max:
            return 'train'
        if year_label in self.val_years:
            return 'val'
        if year_label in self.test_years:
            return 'test'
        return None

    def _scan_fire_events(self, ds, mask):
        """
        Escanea is_fire para encontrar positivos con offset temporal.

        Para cada fuego en t_label:
          - features se extraen en t_features = t_label - 1
          - label = 1 (fuego al dia siguiente)
          - year = year(t_label) para asignar split

        Returns:
            fire_events: list of (t_features, y, x, year_label)
            fire_set: set of (t_features, y, x) para rechazo de negativos
        """
        print("\nEscaneando eventos de fuego (offset t -> t+1)...")
        times = pd.to_datetime(ds.time.values)
        n_times = len(times)

        fire_events = []
        fire_set = set()

        # Read is_fire in large chunks to minimize I/O ops
        SCAN_BATCH = 500
        for batch_start in tqdm(range(0, n_times, SCAN_BATCH),
                                desc="Buscando fuegos",
                                total=(n_times + SCAN_BATCH - 1) // SCAN_BATCH):
            batch_end = min(batch_start + SCAN_BATCH, n_times)
            fire_chunk = ds["is_fire"].isel(
                time=slice(batch_start, batch_end)
            ).values  # shape (batch, y, x)

            for local_t in range(fire_chunk.shape[0]):
                t_label = batch_start + local_t
                if t_label == 0:
                    continue

                year_label = times[t_label].year
                if year_label < self.start_year or year_label > self.end_year:
                    continue

                fire_in_ccaa = (fire_chunk[local_t] > 0) & mask

                if fire_in_ccaa.any():
                    t_features = t_label - 1
                    ys, xs = np.where(fire_in_ccaa)
                    for y, x in zip(ys, xs):
                        fire_events.append((t_features, int(y), int(x), year_label))
                        fire_set.add((t_features, int(y), int(x)))

            del fire_chunk

        print(f"   Total positivos encontrados: {len(fire_events):,}")
        return fire_events, fire_set

    # ──────────────────────────────────────────────────────────────
    # SAMPLE NEGATIVES (per-split, with rejection)
    # ──────────────────────────────────────────────────────────────

    def _sample_negatives_per_split(self, ds, mask, fire_by_split, fire_set):
        """
        Muestrea negativos por split con batch rejection.

        Para cada split:
          n_neg = n_pos * neg_pos_ratio
          Candidate (t, y, x): pixel aleatorio en CCAA, timestep aleatorio del split.
          Rechazado si (t, y, x) in fire_set (habria fuego en t+1).
        """
        print(f"\nMuestreando negativos (ratio {self.neg_pos_ratio}:1 por split)...")

        times = pd.to_datetime(ds.time.values)
        valid_ys, valid_xs = np.where(mask)
        n_pixels = len(valid_ys)

        # Pre-compute valid t_features per split
        valid_t_per_split = defaultdict(list)
        for t in range(len(times) - 1):
            year_label = times[t + 1].year
            if year_label < self.start_year or year_label > self.end_year:
                continue
            split = self._assign_split(year_label)
            if split:
                valid_t_per_split[split].append(t)

        for s in valid_t_per_split:
            valid_t_per_split[s] = np.array(valid_t_per_split[s])

        rng = np.random.default_rng(42)
        neg_by_split = {}

        for split_name in ['train', 'val', 'test']:
            n_pos = len(fire_by_split.get(split_name, []))
            n_neg = int(n_pos * self.neg_pos_ratio)

            if n_neg == 0 or split_name not in valid_t_per_split:
                neg_by_split[split_name] = []
                continue

            t_pool = valid_t_per_split[split_name]
            neg_events = []
            max_attempts = n_neg * 10
            attempts = 0

            with tqdm(total=n_neg, desc=f"  Neg {split_name}") as pbar:
                while len(neg_events) < n_neg and attempts < max_attempts:
                    batch = min(n_neg - len(neg_events), 5000)
                    t_samples = rng.choice(t_pool, size=batch)
                    px_samples = rng.integers(0, n_pixels, size=batch)

                    for i in range(batch):
                        t = int(t_samples[i])
                        y = int(valid_ys[px_samples[i]])
                        x = int(valid_xs[px_samples[i]])
                        year_label = times[t + 1].year

                        if (t, y, x) not in fire_set:
                            neg_events.append((t, y, x, year_label))
                            pbar.update(1)
                            if len(neg_events) >= n_neg:
                                break

                        attempts += 1

            neg_by_split[split_name] = neg_events
            print(f"   {split_name}: {n_pos:,} pos + {len(neg_events):,} neg "
                  f"(ratio {len(neg_events)/n_pos:.2f})" if n_pos > 0 else "")

        return neg_by_split

    # ──────────────────────────────────────────────────────────────
    # EXTRACT FEATURES (memory-safe, by timestep)
    # ──────────────────────────────────────────────────────────────

    def _extract_features(self, ds, points, include_meta=False):
        """
        Extrae features para puntos (t_features, y, x, year_label).

        Optimizaciones:
          - Itera variable-por-variable (no timestep-por-timestep)
          - Spatial crop: solo lee el bounding box de los puntos
          - netCDF4 directo: bypass de xarray para lecturas raw
          - Batches temporales contiguos (~200 steps)

        Args:
            include_meta: si True, anade columnas _meta_* con coordenadas
                          para visualizacion en mapa (solo util para test).
        """
        import netCDF4 as nc4

        print(f"   Extrayendo {len(self.feature_vars)} features "
              f"para {len(points):,} puntos...")

        times = pd.to_datetime(ds.time.values)

        cube_vars = [v for v in self.feature_vars if v not in VIRTUAL_TIME_VARS]
        virtual_vars = [v for v in self.feature_vars if v in VIRTUAL_TIME_VARS]

        # Pre-compute virtual (calendar) vars for all timesteps
        doy = times.dayofyear.values.astype(np.float32)
        two_pi = 2.0 * np.pi
        virtual_cache = {
            'is_weekend': (times.dayofweek.values >= 5).astype(np.float32),
            'day_of_year_sin': np.sin(two_pi * doy / 365.25).astype(np.float32),
            'day_of_year_cos': np.cos(two_pi * doy / 365.25).astype(np.float32),
        }

        t_indices = np.array([p[0] for p in points])
        y_indices = np.array([p[1] for p in points])
        x_indices = np.array([p[2] for p in points])
        year_labels = np.array([p[3] for p in points])

        # Separate static vs dynamic datacube vars
        static_vars = [v for v in cube_vars if v in ds and "time" not in ds[v].dims]
        dynamic_vars = [v for v in cube_vars if v in ds and "time" in ds[v].dims]
        missing_vars = [v for v in cube_vars
                        if v not in ds.data_vars and v not in ds.coords]

        if missing_vars:
            print(f"   Variables no encontradas (skip): {missing_vars}")

        data = {}

        # Static: load once (single 2D array, small)
        for var in static_vars:
            vals = ds[var].values
            data[var] = vals[y_indices, x_indices].astype(np.float32)

        # ── Dynamic: optimized extraction ──
        # 1. Spatial crop: only read the bounding box of our points
        # 2. netCDF4 raw reads: bypass xarray overhead
        # 3. Contiguous time batches: sequential disk access
        BATCH_T = 200
        n_times = len(times)

        y_min, y_max = int(y_indices.min()), int(y_indices.max()) + 1
        x_min, x_max = int(x_indices.min()), int(x_indices.max()) + 1
        y_local = y_indices - y_min
        x_local = x_indices - x_min

        spatial_pct = 100.0 * (y_max - y_min) * (x_max - x_min) / (
            ds.sizes['y'] * ds.sizes['x'])
        print(f"   Crop espacial: y[{y_min}:{y_max}], x[{x_min}:{x_max}] "
              f"({spatial_pct:.0f}% del grid)")

        # Pre-compute batch groupings (reused for every variable)
        batch_keys = t_indices // BATCH_T
        unique_batches = np.unique(batch_keys)

        batch_groups = {}
        for bk in unique_batches:
            bmask = batch_keys == bk
            batch_groups[int(bk)] = (
                np.where(bmask)[0],                                 # positions in output
                (t_indices[bmask] - int(bk) * BATCH_T).astype(int), # local time offset
                y_local[bmask],
                x_local[bmask],
            )

        print(f"   {len(dynamic_vars)} vars dinamicas, "
              f"{len(unique_batches)} bloques de ~{BATCH_T} timesteps")

        # Open raw netCDF4 handle (much faster than xarray for bulk reads)
        nc = nc4.Dataset(str(self.datacube_path), 'r')

        for var_name in tqdm(dynamic_vars, desc="   Extrayendo variables"):
            ncvar = nc.variables[var_name]
            arr = np.empty(len(points), dtype=np.float32)

            for bk, (pos, lt, ys, xs) in batch_groups.items():
                t_start = bk * BATCH_T
                t_end = min(t_start + BATCH_T, n_times)

                # Read with spatial crop: (batch, y_range, x_range)
                raw = ncvar[t_start:t_end, y_min:y_max, x_min:x_max]
                chunk = np.asarray(raw, dtype=np.float32)
                if hasattr(raw, 'mask'):
                    chunk[raw.mask] = np.nan
                arr[pos] = chunk[lt, ys, xs]
                del chunk, raw

            data[var_name] = arr

        nc.close()

        # Virtual (calendar) vars
        for var in virtual_vars:
            if var in virtual_cache:
                data[var] = virtual_cache[var][t_indices]

        df = pd.DataFrame(data)

        # NaN: fill with 0 (tree models handle missing well; few NaN expected)
        feature_cols = [c for c in self.feature_vars if c in df.columns]
        n_nan = df[feature_cols].isna().sum().sum()
        if n_nan > 0:
            print(f"   {n_nan} NaN encontrados, rellenando con 0")
            df[feature_cols] = df[feature_cols].fillna(0)

        # Metadata columns only for test (map visualization).
        # Prefixed _meta_ so get_features_labels() never includes them.
        if include_meta:
            df['_meta_t_features'] = t_indices
            df['_meta_t_label'] = t_indices + 1
            df['_meta_y_idx'] = y_indices
            df['_meta_x_idx'] = x_indices
            df['_meta_year'] = year_labels
            if 'x' in ds.coords:
                df['_meta_x'] = ds.x.values[x_indices]
            if 'y' in ds.coords:
                df['_meta_y'] = ds.y.values[y_indices]
            df['_meta_time_label'] = times[t_indices + 1].values

        gc.collect()
        return df

    # ──────────────────────────────────────────────────────────────
    # BUILD DATASET (main pipeline)
    # ──────────────────────────────────────────────────────────────

    def build_dataset(self):
        """
        Pipeline completo: scan fires -> sample negatives -> extract features -> save.

        - Offset temporal: features en t, label is_fire en t+1.
        - Ratio neg/pos controlado independientemente por split.
        - Rechazo de negativos que coincidan con coordenadas de fuego.
        - Cache: si los parquets existen con los mismos params, no reconstruye.
        """
        if self._check_cache_valid():
            print("Datos existentes con mismos parametros. Saltando build.")
            print(f"   Para forzar rebuild, elimina: {self.params_path}")
            return

        # 1. Open datacube (lazy, no carga nada)
        print("Abriendo datacube...")
        ds = xr.open_dataset(self.datacube_path)

        # 2. CCAA spatial mask
        print("Construyendo mascara espacial...")
        mask = self._build_ccaa_mask(ds)

        # 3. Scan fire events (with t -> t+1 offset)
        fire_events, fire_set = self._scan_fire_events(ds, mask)

        if not fire_events:
            print("No se encontraron fuegos en el rango. Abortando.")
            ds.close()
            return

        # 4. Group fires by split
        fire_by_split = defaultdict(list)
        for event in fire_events:
            split = self._assign_split(event[3])  # year_label
            if split:
                fire_by_split[split].append(event)

        print("\nFuegos por split:")
        for s in ['train', 'val', 'test']:
            print(f"   {s}: {len(fire_by_split[s]):,}")

        # 5. Sample negatives per split (with rejection)
        neg_by_split = self._sample_negatives_per_split(
            ds, mask, fire_by_split, fire_set
        )

        # 6. Extract features and save per split
        print("\nExtrayendo features y guardando splits...")
        for split_name in ['train', 'val', 'test']:
            pos = fire_by_split[split_name]
            neg = neg_by_split[split_name]

            if not pos and not neg:
                print(f"   {split_name}: vacio, skip")
                continue

            all_points = pos + neg
            labels = [1] * len(pos) + [0] * len(neg)

            print(f"\n--- {split_name.upper()} ---")
            df = self._extract_features(
                ds, all_points, include_meta=(split_name == 'test')
            )
            df['is_fire'] = labels

            # Shuffle
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            path = getattr(self, f'{split_name}_file')
            df.to_parquet(path, compression='snappy')
            print(f"   Guardado: {path} ({len(df):,} rows)")

        # 7. Save params for cache validation
        self._save_params()

        ds.close()
        gc.collect()

        print("\nBuild completado.")

    # ──────────────────────────────────────────────────────────────
    # NORMALIZE
    # ──────────────────────────────────────────────────────────────

    def normalize_data(self):
        """
        Normaliza features continuas con StandardScaler.
        Scaler ajustado SOLO en train (sin data leakage).
        Variables binarias excluidas.
        """
        print("\nNormalizando features...")

        binary_vars = {
            'is_waterbody', 'is_holiday',
            'is_weekend', 'is_natura2000'
        }

        df_train = pd.read_parquet(self.train_file)
        cols_to_norm = [
            c for c in self.feature_vars
            if c in df_train.columns and c not in binary_vars
        ]

        print(f"   {len(cols_to_norm)} continuas a normalizar, "
              f"{len(binary_vars)} binarias sin tocar")

        scaler = StandardScaler()
        scaler.fit(df_train[cols_to_norm])

        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"   Scaler guardado: {self.scaler_path}")

        for name, path in [("train", self.train_file),
                           ("val", self.val_file),
                           ("test", self.test_file)]:
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            df[cols_to_norm] = scaler.transform(df[cols_to_norm])
            norm_path = path.replace(".parquet", "_normalized.parquet")
            df.to_parquet(norm_path, compression="snappy")
            print(f"   {name.upper()} -> {norm_path}")

    # ──────────────────────────────────────────────────────────────
    # GET FEATURES / LABELS
    # ──────────────────────────────────────────────────────────────

    def get_features_labels(self, split="train", normalized=True):
        """
        Carga X (features) y y (label) para un split.

        Retorna SOLO las columnas de DEFAULT_FEATURE_VARS como features.
        Evita location leakage (no incluye x, y, year, etc.).

        Args:
            split: "train", "val" o "test".
            normalized: si True, carga la version normalizada.

        Returns:
            (X, y, feature_names) o (None, None, None) si no existe.
        """
        suffix = "_normalized" if normalized else ""
        path = os.path.join(self.output_dir, f"{split}{suffix}.parquet")

        if not os.path.exists(path):
            print(f"Archivo no encontrado: {path}")
            return None, None, None

        df = pd.read_parquet(path)

        y = df['is_fire']
        available = [c for c in self.feature_vars if c in df.columns]
        X = df[available]

        return X, y, available

    def get_metadata(self, split="test"):
        """
        Carga metadatos espaciotemporales de un split para visualizacion.

        Columnas retornadas (si existen):
          _meta_x, _meta_y     : coordenadas reales (proyeccion del datacube)
          _meta_y_idx, _meta_x_idx : indices de grid
          _meta_time_label     : datetime del dia del fuego (t+1)
          _meta_year           : anio del label
          _meta_t_features     : indice temporal de features
          _meta_t_label        : indice temporal del label
          is_fire              : label real (0/1)

        Args:
            split: "train", "val" o "test".

        Returns:
            DataFrame con columnas _meta_* + is_fire, o None si no existe.
        """
        path = os.path.join(self.output_dir, f"{split}.parquet")

        if not os.path.exists(path):
            print(f"Archivo no encontrado: {path}")
            return None

        df = pd.read_parquet(path)
        meta_cols = [c for c in df.columns if c.startswith('_meta_')]
        meta_cols.append('is_fire')
        return df[[c for c in meta_cols if c in df.columns]]

    # ──────────────────────────────────────────────────────────────
    # METADATA (save)
    # ──────────────────────────────────────────────────────────────

    def save_metadata(self):
        """Guarda metadatos del proceso de generacion."""
        stats = {}
        for name, path in [("train", self.train_file),
                           ("val", self.val_file),
                           ("test", self.test_file)]:
            if os.path.exists(path):
                df = pd.read_parquet(path, columns=['is_fire'])
                n_fire = int((df['is_fire'] == 1).sum())
                n_nofire = int((df['is_fire'] == 0).sum())
                ratio = n_nofire / n_fire if n_fire > 0 else 0
                stats[name] = {
                    'total': len(df),
                    'fire': n_fire,
                    'nofire': n_nofire,
                    'ratio': round(ratio, 2),
                }

        metadata = {
            'creation_date': datetime.now().isoformat(),
            'datacube_path': self.datacube_path,
            'period': f"{self.start_year}-{self.end_year}",
            'ccaa_codes': self.ccaa_target,
            'ccaa_names': [self.CCAA_NAMES.get(c, f"Unknown_{c}")
                           for c in self.ccaa_target],
            'neg_pos_ratio': self.neg_pos_ratio,
            'temporal_offset': 'features(t) -> label(t+1)',
            'train_year_max': self.train_year_max,
            'val_years': sorted(self.val_years),
            'test_years': sorted(self.test_years),
            'features': self.feature_vars,
            'n_features': len(self.feature_vars),
            'split_stats': stats,
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadatos guardados: {self.metadata_path}")


# ══════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO (convenience wrapper)
# ══════════════════════════════════════════════════════════════════

def run_full_pipeline(datacube_path, output_dir="data/processed/tabular",
                      start_year=2009, end_year=2024,
                      ccaa_target=None, neg_pos_ratio=2.0,
                      train_year_max=2020, val_years=(2021, 2022),
                      test_years=(2023, 2024)):
    """Ejecuta build + normalize + metadata en un solo comando."""
    dataset = SpainCubeFireDataset(
        datacube_path=datacube_path,
        output_dir=output_dir,
        start_year=start_year,
        end_year=end_year,
        ccaa_target=ccaa_target,
        neg_pos_ratio=neg_pos_ratio,
        train_year_max=train_year_max,
        val_years=val_years,
        test_years=test_years,
    )

    dataset.build_dataset()
    dataset.normalize_data()
    dataset.save_metadata()

    return dataset


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline tabular para XGBoost / Random Forest"
    )
    parser.add_argument("--datacube", type=str,
                        default=str(PROJECT_ROOT / "data" / "IberFire.nc"))
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "tabular"))
    parser.add_argument("--start_year", type=int, default=2009)
    parser.add_argument("--end_year", type=int, default=2024)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--train_max", type=int, default=2020)
    parser.add_argument("--val_years", nargs="+", type=int, default=[2021, 2022])
    parser.add_argument("--test_years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--ccaa", nargs="+", type=int,
                        default=[12, 3, 7, 11, 10, 1])
    args = parser.parse_args()

    run_full_pipeline(
        datacube_path=args.datacube,
        output_dir=args.output,
        start_year=args.start_year,
        end_year=args.end_year,
        ccaa_target=args.ccaa,
        neg_pos_ratio=args.neg_ratio,
        train_year_max=args.train_max,
        val_years=tuple(args.val_years),
        test_years=tuple(args.test_years),
    )
