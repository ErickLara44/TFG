"""
prepare_spread_patches_v2.py — Dataset SOLO con propagación real (y_sum > 0)
=============================================================================
Diferencias respecto a prepare_spread_patches.py (v1):
  - Filtra y DESCARTA patches donde y_sum == 0 (fuego no se propagó ese día)
  - Guarda SOLO los días donde el fuego avanzó: son los útiles para aprender
  - Usa el mismo split temporal y augmentaciones con corrección de viento

Resultado: dataset más pequeño pero donde CADA muestra enseña propagación real.
  → Elimina el mínimo local trivial de predecir-cero-siempre.

Output dir: data/processed/patches/spread_224_fires_only/
"""

import os
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import argparse
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data.data_prop_improved import (
    SpreadDataset,
    generate_temporal_splits,
    DEFAULT_FEATURE_VARS as CHANNELS,
)
from torch.utils.data import DataLoader


def save_patches_fires_only(dataset, output_dir, split_name):
    output_dir = Path(output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)



    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── Lógica de reanudación: detectar cuántos originales ya existen ──
    existing_originals = sorted(output_dir.glob("sample_*_orig.pt"))
    already_saved = len(existing_originals)
    if already_saved > 0:
        print(f"⏩ Reanudando: {already_saved} muestras ya guardadas, saltando...")

    print(f"🚀 Generando parches SOLO PROPAGACIÓN para split: {split_name}")
    print(f"   Total muestras fuente: {len(dataset)}")

    saved, skipped = 0, 0

    for i, (x_batch, y_batch) in tqdm(enumerate(loader), total=len(loader),
                                       desc=f"Saving {split_name}"):
        try:
            x_orig = x_batch[0]
            y_orig = y_batch[0]

            # ── FILTRO CLAVE: solo guardar si hay propagación real ──
            if y_orig.sum().item() == 0:
                skipped += 1
                continue

            # ── REANUDACIÓN: saltar muestras ya guardadas ──
            if saved < already_saved:
                orig_path = output_dir / f"sample_{saved:06d}_orig.pt"
                if orig_path.exists():
                    saved += 1
                    continue  # ya guardado, pasar al siguiente

            # Sin augmentación offline — solo guardamos el original
            # (la augmentación se aplica on-the-fly en el DataLoader durante entrenamiento)
            sample_id = f"sample_{saved:06d}_orig.pt"
            torch.save({'x': x_orig, 'y': y_orig}, output_dir / sample_id)

            saved += 1

        except Exception as e:
            print(f"⚠️ Error en muestra {i}: {e}")

    print(f"   ✅ Guardados: {saved} | Descartados (y=0): {skipped}")
    return saved


def save_negatives(dataset, output_dir, neg_ratio=0.5):
    """
    Guarda muestras donde y_sum == 0 (fuego presente pero sin propagación).
    Sirven como hard negatives para entrenar al modelo a NO predecir fuego.
    Se guardan en output_dir/train_neg/ con hasta neg_ratio * n_positivos muestras.
    """
    pos_dir = Path(output_dir) / 'train'
    n_positives = len(sorted(pos_dir.glob('sample_*_orig.pt')))
    max_neg = int(n_positives * neg_ratio)

    neg_dir = Path(output_dir) / 'train_neg'
    neg_dir.mkdir(parents=True, exist_ok=True)
    already_saved = len(sorted(neg_dir.glob('sample_*_orig.pt')))

    if already_saved >= max_neg:
        print(f"⏩ Negativos ya generados: {already_saved}/{max_neg}, saltando.")
        return already_saved

    if already_saved > 0:
        print(f"⏩ Reanudando negativos: {already_saved}/{max_neg} ya guardados...")

    print(f"🟥 Generando negativos (y=0) para train: máximo {max_neg} muestras")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    saved = already_saved
    pos_seen = 0  # pares ya guardados como positivos (para sincronizar reanudación)

    for i, (x_batch, y_batch) in tqdm(enumerate(loader), total=len(loader),
                                       desc="Saving train_neg"):
        if saved >= max_neg:
            break
        try:
            x_orig = x_batch[0]
            y_orig = y_batch[0]

            if y_orig.sum().item() != 0:
                pos_seen += 1
                continue  # es positivo, no nos interesa aquí

            # Es negativo — ¿ya guardado?
            if saved < already_saved:
                neg_path = neg_dir / f"sample_{saved:06d}_orig.pt"
                if neg_path.exists():
                    saved += 1
                    continue

            torch.save({'x': x_orig, 'y': y_orig}, neg_dir / f"sample_{saved:06d}_orig.pt")
            saved += 1

        except Exception as e:
            print(f"⚠️ Error en muestra {i}: {e}")

    print(f"   ✅ Negativos guardados: {saved}")
    return saved


def save_all_samples(dataset, output_dir, split_name, crop_size=32):
    """
    Guarda TODAS las muestras (y_sum>0 Y y_sum==0) croppadas a crop_size×crop_size
    centradas en el fuego. Genera el dataset completo para que el modelo aprenda
    tanto cuándo el fuego se propaga como cuándo no.
    La etiqueta 'is_positive' permite aplicar WeightedRandomSampler en train.
    """
    out_dir = Path(output_dir) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(sorted(out_dir.glob("sample_*_orig.pt")))
    if existing > 0:
        print(f"⏩ Reanudando: {existing} muestras ya guardadas...")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    saved, n_pos, n_neg = existing, 0, 0
    half = crop_size // 2
    cx = cy = dataset[0][0].shape[-1] // 2 if hasattr(dataset, '__getitem__') else 112

    print(f"📦 Guardando TODAS las muestras (pos+neg) — crop {crop_size}×{crop_size} — split: {split_name}")
    print(f"   Total muestras fuente: {len(dataset)}")

    for i, (x_batch, y_batch) in tqdm(enumerate(loader), total=len(loader),
                                       desc=f"Saving {split_name} (all)"):
        if i < existing:
            continue  # reanudación
        try:
            x = x_batch[0]  # (T, C, H, W)
            y = y_batch[0]  # (1, H, W)

            H, W = x.shape[-2], x.shape[-1]
            h0 = max(0, H // 2 - half)
            w0 = max(0, W // 2 - half)
            x_crop = x[..., h0:h0+crop_size, w0:w0+crop_size]
            y_crop = y[..., h0:h0+crop_size, w0:w0+crop_size]

            is_pos = (y_crop.sum().item() > 0)
            if is_pos: n_pos += 1
            else:       n_neg += 1

            torch.save({'x': x_crop, 'y': y_crop, 'is_positive': is_pos},
                       out_dir / f"sample_{saved:06d}_orig.pt")
            saved += 1

        except Exception as e:
            print(f"⚠️ Error en muestra {i}: {e}")

    print(f"   ✅ Total guardados: {saved} (positivos: {n_pos} | negativos: {n_neg})")
    return saved, n_pos, n_neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size",        type=int, default=224)
    parser.add_argument("--output_dir",       type=str,
                        default="data/processed/patches/spread_224_fires_only")
    parser.add_argument("--temporal_context", type=int, default=3)
    parser.add_argument("--splits",           type=str, nargs='+',
                        default=['train', 'val', 'test'],
                        help="Splits a generar (ej: --splits val test)")
    parser.add_argument("--neg_ratio",        type=float, default=0.0,
                        help="Ratio neg/pos para generar negativos en train_neg/ (ej: 0.5)")
    parser.add_argument("--all_samples",      action='store_true',
                        help="Guardar TODAS las muestras (y_sum=0 incluido) en nuevo dataset")
    parser.add_argument("--save_crop_size",   type=int, default=32,
                        help="Crop size para guardar en --all_samples mode")
    args = parser.parse_args()

    print(f"📂 Cargando Datacube: {config.DATACUBE_PATH}")
    datacube = xr.open_dataset(config.DATACUBE_PATH)

    print("✂️ Generando splits temporal...")
    splits = generate_temporal_splits(datacube, strict=True)

    if args.all_samples:
        # Modo nuevo: guarda TODAS las muestras (pos+neg) a crop_size=save_crop_size
        # Nuevo output dir: spread_32_all/
        all_output = args.output_dir.replace("spread_224_fires_only", "spread_32_all")
        if "spread_224" not in args.output_dir:
            all_output = args.output_dir + "_all"
        print(f"\n🚀 Modo --all_samples: guardando pos+neg a {all_output} (crop {args.save_crop_size}×{args.save_crop_size})")
        for split_name in args.splits:
            if split_name not in splits:
                continue
            indices = splits[split_name]
            ds = SpreadDataset(
                datacube, indices,
                temporal_context=args.temporal_context,
                filter_fire_samples=True,
                preload_ram=False,
                crop_size=args.crop_size,  # 224 para cargar, luego cropeamos a save_crop_size
                feature_vars=CHANNELS
            )
            save_all_samples(ds, all_output, split_name, crop_size=args.save_crop_size)
        print(f"\n✅ Dataset completo (pos+neg) generado en: {all_output}")
    else:
        for split_name in args.splits:
            if split_name not in splits:
                continue
            indices = splits[split_name]
            ds = SpreadDataset(
                datacube, indices,
                temporal_context=args.temporal_context,
                filter_fire_samples=True,
                preload_ram=False,
                crop_size=args.crop_size,
                feature_vars=CHANNELS
            )
            save_patches_fires_only(ds, args.output_dir, split_name)

        # Generar negativos si se pide
        if args.neg_ratio > 0 and 'train' in args.splits:
            train_indices = splits['train']
            train_ds = SpreadDataset(
                datacube, train_indices,
                temporal_context=args.temporal_context,
                filter_fire_samples=True,
                preload_ram=False,
                crop_size=args.crop_size,
                feature_vars=CHANNELS
            )
            save_negatives(train_ds, args.output_dir, neg_ratio=args.neg_ratio)

        print("\n✅ Nuevo dataset (fires_only) generado en:", args.output_dir)




if __name__ == "__main__":
    main()
