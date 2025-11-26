import torch
from torch.utils.data import Dataset
import numpy as np

# 🔹 Variables fijas que quieres usar
DEFAULT_FEATURE_VARS = [
    'elevation_mean','slope_mean',
    'CLC_2018_forest_proportion','CLC_2018_scrub_proportion','CLC_2018_agricultural_proportion',
    'dist_to_roads_mean','popdens_2018','is_waterbody',
    't2m_mean','RH_min','wind_speed_mean','total_precipitation_mean',
    'NDVI','SWI_010','FWI','LST','is_near_fire','wind_direction_mean'
]

class SpreadDataset(Dataset):
    """
    Dataset para predicción de propagación espacial.
    🔥 Incluye estado actual del fuego como contexto esencial
    """
    def __init__(self, datacube, indices, temporal_context=3, include_fire_state=True, 
                 filter_fire_samples=True, min_fire_pixels=5, feature_vars=None):
        self.datacube = datacube
        self.indices = indices
        self.temporal_context = temporal_context
        self.include_fire_state = include_fire_state
        self.filter_fire_samples = filter_fire_samples
        self.min_fire_pixels = min_fire_pixels

        # ✅ Usar tus variables fijas
        self.feature_vars = feature_vars if feature_vars is not None else DEFAULT_FEATURE_VARS

        # Validar que existen en el datacube
        missing = [v for v in self.feature_vars if v not in datacube.data_vars]
        if missing:
            raise ValueError(f"❌ Variables no encontradas en datacube: {missing}")

        # Filtrar muestras que tienen fuego actual (para propagación)
        if filter_fire_samples:
            self.indices = self._filter_samples_with_fire()
        
        print(f"🔥 SpreadDataset inicializado:")
        print(f"   Variables: {len(self.feature_vars)} + {'1 (fire_state)' if include_fire_state else '0'}")
        print(f"   Contexto temporal: {temporal_context} días")
        print(f"   Muestras totales: {len(self.indices)}")
        print(f"   Filtrado por fuego: {filter_fire_samples}")

    def _filter_samples_with_fire(self):
        """Filtra muestras que tienen fuego en al menos uno de los timesteps de contexto"""
        valid_indices = []
        for sample_info in self.indices:
            t0 = sample_info["time_index"]
            has_fire = False
            for dt in range(self.temporal_context):
                t = t0 - (self.temporal_context - 1 - dt)
                if 0 <= t < len(self.datacube.time):
                    fire_map = self.datacube["is_fire"].isel(time=t).values
                    if np.sum(fire_map) >= self.min_fire_pixels:
                        has_fire = True
                        break
            if has_fire:
                valid_indices.append(sample_info)
        
        print(f"   📊 Muestras con fuego: {len(valid_indices)}/{len(self.indices)} "
              f"({len(valid_indices)/len(self.indices)*100:.1f}%)")
        return valid_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_info = self.indices[idx]
        t0 = sample_info["time_index"]

        # --- 1) Secuencia temporal con estado del fuego ---
        x_seq = []
        for dt in range(self.temporal_context):
            t = t0 - (self.temporal_context - 1 - dt)
            t = max(0, min(t, len(self.datacube.time) - 1))  # clamp

            channels = []
            # Variables ambientales/meteorológicas
            for var in self.feature_vars:
                arr = self.datacube[var].isel(time=t).values.astype(np.float32)
                channels.append(arr)

            # 🔥 Estado actual del fuego
            if self.include_fire_state:
                fire_state = self.datacube["is_fire"].isel(time=t).values.astype(np.float32)
                channels.append(fire_state)

            x_seq.append(np.stack(channels, axis=0))  # (C+1, H, W)
        
        x_seq = np.stack(x_seq, axis=0)  # (T, C+1, H, W)

        # --- 2) Etiqueta: mapa de fuego en t+1 ---
        t_next = min(t0 + 1, len(self.datacube.time) - 1)
        y_map = self.datacube["is_fire"].isel(time=t_next).values.astype(np.float32)
        y_map = np.expand_dims(y_map, axis=0)  # (1, H, W)

        return torch.FloatTensor(x_seq), torch.FloatTensor(y_map)

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

print("✅ SpreadDataset mejorado implementado!")
print("🚀 Usa setup_datasets_example(datacube) para configurar todo")