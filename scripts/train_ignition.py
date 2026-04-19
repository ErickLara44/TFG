"""
Script para entrenar el ConvLSTM de ignicion.
Fase 1: Precomputa parches a disco (.pt) — se hace una sola vez.
Fase 2: Entrena desde los .pt — rapido en cada epoch.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader

from src import config
from src.data.data_ignition_improved import (
    DEFAULT_FEATURE_VARS,
    create_ignition_datasets,
    precompute_patches,
    PrecomputedIgnitionDataset,
)
from src.data.data_prop_improved import create_year_split
from src.models.ignition import RobustFireIgnitionModel, train_robust_ignition_model


PATCHES_DIR = Path(config.DATA_DIR) / "processed" / "ignition_patches"


def phase1_precompute():
    """Genera todos los parches .pt desde el datacube (una sola vez)."""
    print("=" * 60)
    print("FASE 1: PRECOMPUTAR PARCHES")
    print("=" * 60)

    temporal_context = config.MODEL_CONFIG['temporal_context']

    print(f"\n1) Cargando datacube desde {config.DATACUBE_PATH} ...")
    datacube = xr.open_dataset(config.DATACUBE_PATH)
    print(f"   Dimensiones: {dict(datacube.sizes)}")

    train_years = list(range(2015, 2022))
    val_years = [2022, 2023]
    test_years = [2024]

    print(f"\n2) Generando splits por anios...")
    splits = create_year_split(
        datacube, train_years, val_years, test_years,
        min_temporal_context=temporal_context
    )

    for k, v in splits.items():
        print(f"   {k}: {len(v)} timesteps disponibles")

    # samples altos para capturar todos los fuegos disponibles
    print(f"\n3) Creando datasets de ignicion (todos los fuegos)...")
    ignition_datasets = create_ignition_datasets(
        datacube, splits,
        temporal_context=temporal_context,
        patch_size=64,
        samples_train=50000,  # cap alto, usara todos los fuegos reales
        samples_val=20000,
        samples_test=20000,
        max_fires_per_day=10,
    )

    # Precomputar cada split
    for split_name in ['train', 'val', 'test']:
        ds = ignition_datasets[split_name]
        split_dir = PATCHES_DIR / split_name
        print(f"\n   Precomputando '{split_name}': {len(ds)} parches -> {split_dir}")
        saved = precompute_patches(ds, str(split_dir), start_index=0)
        print(f"   Guardados: {saved}")

    print("\n" + "=" * 60)
    print("PRECOMPUTACION COMPLETADA")
    print("=" * 60)


def phase2_train():
    """Entrena desde parches precomputados (.pt) — rapido."""
    print("=" * 60)
    print("FASE 2: ENTRENAMIENTO DESDE PARCHES PRECOMPUTADOS")
    print("=" * 60)

    temporal_context = config.MODEL_CONFIG['temporal_context']
    epochs = config.TRAINING_CONFIG['epochs']
    lr = config.TRAINING_CONFIG['learning_rate']
    device = config.TRAINING_CONFIG['device']
    batch_size = 16
    num_workers = 4
    n_channels = len(DEFAULT_FEATURE_VARS)

    print(f"  Canales (features)   : {n_channels}")
    print(f"  Temporal context     : {temporal_context}")
    print(f"  Epochs               : {epochs}")
    print(f"  Batch size           : {batch_size}")
    print(f"  Learning rate        : {lr}")
    print(f"  Device               : {device}")

    # Cargar datasets precomputados
    datasets = {}
    for split_name in ['train', 'val', 'test']:
        split_dir = PATCHES_DIR / split_name
        n_patches = len(list(split_dir.glob("patch_*.pt")))
        print(f"  {split_name}: {n_patches} parches en {split_dir}")
        datasets[split_name] = PrecomputedIgnitionDataset(
            patches_dir=str(split_dir),
            indices=list(range(n_patches)),
            mode="convlstm",
            augment=(split_name == 'train'),
        )

    train_loader = DataLoader(
        datasets['train'], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        datasets['val'], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    # Verificar un batch
    print("\n  Verificando primer batch...")
    sample_x, sample_y = next(iter(train_loader))
    print(f"  Batch shape: x={sample_x.shape}, y={sample_y.shape}")
    print(f"  Labels: fire={int((sample_y > 0.5).sum())}, no_fire={int((sample_y <= 0.5).sum())}")

    # Crear modelo
    print(f"\n  Creando modelo RobustFireIgnitionModel...")
    model = RobustFireIgnitionModel(
        num_input_channels=n_channels,
        temporal_context=temporal_context,
        hidden_dims=[64, 128],
        dropout=0.2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros totales: {total_params:,}")

    # Entrenar
    print(f"\n  Iniciando entrenamiento ({epochs} epochs)...")
    save_path = str(config.OUTPUTS_DIR / "ignition_metrics.json")

    history = train_robust_ignition_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_metrics_path=save_path,
        early_stopping_patience=10,
        pos_weight=2.0,
    )

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor F1: {max(history['f1']):.4f}")
    print(f"  Mejor AUROC: {max(history['auroc']):.4f}")
    print(f"  Modelo guardado en: best_robust_ignition_model.pth")
    print(f"  Metricas guardadas en: {save_path}")
    print("=" * 60)


def main():
    train_dir = PATCHES_DIR / "train"
    patches_exist = train_dir.exists() and len(list(train_dir.glob("patch_*.pt"))) > 0

    if not patches_exist:
        print("No se encontraron parches precomputados.")
        print("Ejecutando FASE 1 (precomputacion)...\n")
        phase1_precompute()
        print("\nParches listos. Ejecutando FASE 2 (entrenamiento)...\n")
        phase2_train()
    else:
        n = len(list(train_dir.glob("patch_*.pt")))
        print(f"Encontrados {n} parches precomputados en {train_dir}")
        print("Saltando FASE 1, directo a FASE 2 (entrenamiento).\n")
        phase2_train()


if __name__ == "__main__":
    main()
