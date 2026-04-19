
import os
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data.data_prop_improved import (
    SpreadDataset,
    generate_temporal_splits,
    DEFAULT_FEATURE_VARS as CHANNELS,
)

from torch.utils.data import DataLoader, Dataset as TorchDataset

class WorkerCompatibleSpreadDataset(TorchDataset):
    """
    Wrapper para hacer SpreadDataset compatible con multiprocessing (num_workers > 0).
    Abre el Datacube en cada worker independientemente.
    """
    def __init__(self, nc_path, indices, temporal_context=3, filter_fire=True):
        self.nc_path = nc_path
        self.indices = indices
        self.temporal_context = temporal_context
        self.filter_fire = filter_fire
        
        # Internals (Lazy Init)
        self.ds = None
        self.spread_ds = None
        
    def _init_worker(self):
        if self.ds is None:
            # print(f"🔧 Worker inicializando Datacube: {self.nc_path}")
            self.ds = xr.open_dataset(self.nc_path)
            # Instanciamos la lógica original pero con el ds local del worker
            self.spread_ds = SpreadDataset(
                datacube=self.ds,
                indices=self.indices,
                temporal_context=self.temporal_context,
                filter_fire_samples=False, # Ya vienen filtrados en indices
                preload_ram=False
            )
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        self._init_worker()
        # Delegamos en la lógica original
        # SpreadDataset[idx] usa self.indices[idx]
        return self.spread_ds[idx]

def save_patches(dataset, output_dir, split_name):
    output_dir = Path(output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extraemos info necesaria para reconstruir el dataset en workers
    nc_path = config.DATACUBE_PATH
    indices = dataset.indices
    
    # DataLoader con Sequential Execution (Estabilidad > Velocidad)
    # Evitamos multiprocessing para no saturar descriptores de archivo ni chocar con OpenMP
    loader = DataLoader(
        dataset, # Usamos dataset directo (ya tiene el datacube abierto)
        batch_size=1, 
        shuffle=False, 
        num_workers=0, # 0 = Hilo principal
        collate_fn=None 
    )
    
    print(f"🚀 Generando parches para split: {split_name} en {output_dir}")
    print(f"   Total muestras: {len(dataset)}")
    print(f"   Workers: 0 (Secuencial)")

    # Indices de viento para rotación
    try:
        idx_u = CHANNELS.index('wind_u')
        idx_v = CHANNELS.index('wind_v')
    except ValueError:
        idx_u, idx_v = -1, -1

    for i, (x_batch, y_batch) in tqdm(enumerate(loader), total=len(loader), desc=f"Saving {split_name}"):
        try:
            # Desempaquetar batch (size 1)
            x_orig = x_batch[0]
            y_orig = y_batch[0]
            
            # --- AUGMENTATION LOGIC ---
            transformations = []
            
            if split_name == 'train':
                transformations = [
                    ('orig', lambda x, y: (x, y)),
                    ('rot90', lambda x, y: (torch.rot90(x, 1, [2, 3]), torch.rot90(y, 1, [1, 2]))),
                    ('rot180', lambda x, y: (torch.rot90(x, 2, [2, 3]), torch.rot90(y, 2, [1, 2]))),
                    ('rot270', lambda x, y: (torch.rot90(x, 3, [2, 3]), torch.rot90(y, 3, [1, 2]))),
                    ('flipH', lambda x, y: (torch.flip(x, [3]), torch.flip(y, [2]))),
                    ('flipV', lambda x, y: (torch.flip(x, [2]), torch.flip(y, [1]))),
                    ('flipH_rot90', lambda x, y: (torch.rot90(torch.flip(x, [3]), 1, [2, 3]), torch.rot90(torch.flip(y, [2]), 1, [1, 2]))),
                    ('flipV_rot90', lambda x, y: (torch.rot90(torch.flip(x, [2]), 1, [2, 3]), torch.rot90(torch.flip(y, [1]), 1, [1, 2]))),
                ]
            else:
                transformations = [('orig', lambda x, y: (x, y))]
            
            for suffix, transform_func in transformations:
                x_aug, y_aug = transform_func(x_orig.clone(), y_orig.clone())
                
                # --- CORRECCIÓN DE VIENTO ---
                if idx_u != -1 and idx_v != -1:
                    u = x_aug[:, idx_u, :, :].clone()
                    v = x_aug[:, idx_v, :, :].clone()
                    
                    if suffix == 'rot90':
                        x_aug[:, idx_u, :, :] = -v
                        x_aug[:, idx_v, :, :] = u
                    elif suffix == 'rot180':
                        x_aug[:, idx_u, :, :] = -u
                        x_aug[:, idx_v, :, :] = -v
                    elif suffix == 'rot270':
                        x_aug[:, idx_u, :, :] = v
                        x_aug[:, idx_v, :, :] = -u
                    elif suffix == 'flipH':
                        x_aug[:, idx_u, :, :] = -u
                    elif suffix == 'flipV':
                        x_aug[:, idx_v, :, :] = -v
                    elif suffix == 'flipH_rot90':
                        x_aug[:, idx_u, :, :] = -v
                        x_aug[:, idx_v, :, :] = -u
                    elif suffix == 'flipV_rot90':
                        x_aug[:, idx_u, :, :] = v
                        x_aug[:, idx_v, :, :] = u
                
                # Guardar
                sample_id = f"sample_{i:06d}_{suffix}.pt"
                torch.save({'x': x_aug, 'y': y_aug}, output_dir / sample_id)
            
        except Exception as e:
            print(f"⚠️ Error en muestra {i}: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="data/processed/patches/spread_224")
    parser.add_argument("--temporal_context", type=int, default=3)
    args = parser.parse_args()
    
    print(f"📂 Cargando Datacube: {config.DATACUBE_PATH}")
    # Usar chunks={} habilita dask y puede mejorar el acceso aleatorio si se configura bien,
    # pero aquí vamos secuencial. Dejamos lazy loading por defecto.
    datacube = xr.open_dataset(config.DATACUBE_PATH)
    
    # Generar splits
    print("✂️ Generando splits temporal...")
    splits = generate_temporal_splits(datacube, strict=True)
    
    # Crear Datasets (sin preload_ram, porque vamos a leer para guardar)
    # Importante: filter_fire_samples=True para solo guardar incendios
    # crop_size=args.crop_size para recortar
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits: continue
        
        indices = splits[split_name]
        ds = SpreadDataset(
            datacube, 
            indices, 
            temporal_context=args.temporal_context, 
            filter_fire_samples=True, 
            preload_ram=False, 
            crop_size=args.crop_size, 
            feature_vars=CHANNELS
        )
        
        save_patches(ds, args.output_dir, split_name)
        
    print("✅ Generación de parches completada.")

if __name__ == "__main__":
    main()
