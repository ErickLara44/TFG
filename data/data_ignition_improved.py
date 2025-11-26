import torch
from torch.utils.data import Dataset
import numpy as np

DEFAULT_FEATURE_VARS = [
    'elevation_mean','slope_mean',
    'CLC_2018_forest_proportion','CLC_2018_scrub_proportion','CLC_2018_agricultural_proportion',
    'dist_to_roads_mean','popdens_2018','is_waterbody',
    't2m_mean','RH_min','wind_speed_mean','total_precipitation_mean',
    'NDVI','SWI_010','FWI','LST','is_near_fire','wind_direction_mean'
]

class IgnitionDataset(Dataset):
    """
    Dataset para predicción de ignición (clasificación binaria).
    🔥 MEJORADO: Con validaciones, estadísticas y mejor manejo de errores
    
    Entrada:
      - x_seq: [T, C, H, W] (variables meteorológicas/topográficas)
    Salida:
      - y: [1] (0=no fuego, 1=fuego en t+1)
    """
    def __init__(self, datacube, indices, temporal_context=3, mode="convlstm", 
                 feature_vars=None, balance_classes=False):
        self.datacube = datacube
        self.indices = indices
        self.temporal_context = temporal_context
        self.mode = mode
        self.feature_vars = feature_vars if feature_vars is not None else DEFAULT_FEATURE_VARS
        self.balance_classes = balance_classes
        
        # Validar que las variables existen en el datacube
        self._validate_feature_vars()
        
        # Filtrar índices válidos
        self.indices = self._filter_valid_indices()
        
        # Balancear clases si se solicita
        if balance_classes:
            self.indices = self._balance_fire_nofire_samples()
        
        print(f"🔥 IgnitionDataset inicializado:")
        print(f"   Variables: {len(self.feature_vars)}")
        print(f"   Contexto temporal: {temporal_context} días")
        print(f"   Muestras válidas: {len(self.indices)}")
        print(f"   Balanceado: {balance_classes}")

    def _validate_feature_vars(self):
        """Valida que todas las variables existen en el datacube"""
        missing_vars = []
        for var in self.feature_vars:
            if var not in self.datacube.data_vars:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Variables faltantes en datacube: {missing_vars}")
            print(f"📊 Variables disponibles: {list(self.datacube.data_vars)}")
            raise ValueError(f"Variables faltantes: {missing_vars}")
        
        print(f"✅ Todas las variables encontradas en datacube")

    def _filter_valid_indices(self):
        """Filtra índices que tienen suficiente contexto temporal"""
        valid_indices = []
        total_times = len(self.datacube.time)
        
        for sample_info in self.indices:
            t0 = sample_info["time_index"]
            
            # Verificar que tenemos suficiente contexto temporal hacia atrás
            min_t = t0 - (self.temporal_context - 1)
            # Verificar que podemos obtener la etiqueta en t+1
            max_t = t0 + 1
            
            if min_t >= 0 and max_t < total_times:
                valid_indices.append(sample_info)
        
        print(f"   📊 Índices válidos: {len(valid_indices)}/{len(self.indices)} "
              f"({len(valid_indices)/len(self.indices)*100:.1f}%)")
        
        return valid_indices

    def _balance_fire_nofire_samples(self):
        """Balancea muestras con/sin fuego para evitar desbalance de clases"""
        fire_samples = []
        no_fire_samples = []
        
        print("⚖️ Balanceando clases...")
        
        for sample_info in self.indices:
            t0 = sample_info["time_index"]
            t_next = t0 + 1
            
            y_map = self.datacube["is_fire"].isel(time=t_next).values
            has_fire = 1 if np.sum(y_map) > 0 else 0
            
            if has_fire:
                fire_samples.append(sample_info)
            else:
                no_fire_samples.append(sample_info)
        
        # Balancear usando el mínimo entre ambas clases
        min_samples = min(len(fire_samples), len(no_fire_samples))
        balanced_samples = fire_samples[:min_samples] + no_fire_samples[:min_samples]
        
        # Mezclar aleatoriamente
        np.random.shuffle(balanced_samples)
        
        print(f"   🔥 Muestras con fuego: {len(fire_samples)}")
        print(f"   💧 Muestras sin fuego: {len(no_fire_samples)}")
        print(f"   ⚖️ Muestras balanceadas: {len(balanced_samples)} "
              f"({min_samples} por clase)")
        
        return balanced_samples

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_info = self.indices[idx]
        t0 = sample_info["time_index"]

        # --- 1) Construir la secuencia temporal ---
        x_seq = []
        for dt in range(self.temporal_context):
            t = t0 - (self.temporal_context - 1 - dt)
            
            # Validación adicional (por seguridad)
            if t < 0:
                t = 0
            elif t >= len(self.datacube.time):
                t = len(self.datacube.time) - 1
            
            channels = []
            for var in self.feature_vars:
                arr = self.datacube[var].isel(time=t).values.astype(np.float32)
                channels.append(arr)
            
            x_seq.append(np.stack(channels, axis=0))  # (C,H,W)
        
        x_seq = np.stack(x_seq, axis=0)  # (T,C,H,W)

        # --- 2) Etiqueta: fuego en t+1 ---
        t_next = t0 + 1
        if t_next >= len(self.datacube.time):
            t_next = len(self.datacube.time) - 1
            
        y_map = self.datacube["is_fire"].isel(time=t_next).values.astype(np.float32)
        y_scalar = 1.0 if np.sum(y_map) > 0 else 0.0

        # --- 3) Ajustar formato según el modo ---
        if self.mode == "cnn":
            # CNN: (T*C, H, W)
            T, C, H, W = x_seq.shape
            x_out = x_seq.reshape(T*C, H, W)
        else:
            # ConvLSTM: (T,C,H,W)
            x_out = x_seq

        return torch.FloatTensor(x_out), torch.FloatTensor([y_scalar])

    def get_feature_info(self):
        """Información sobre las variables de entrada"""
        print("🔍 INFORMACIÓN DE VARIABLES:")
        for i, var in enumerate(self.feature_vars):
            print(f"   Canal {i:2d}: {var}")
        
        return self.feature_vars

    def get_class_distribution(self, n_samples=None):
        """Analiza la distribución de clases en el dataset"""
        if n_samples is None:
            n_samples = len(self.indices)
        else:
            n_samples = min(n_samples, len(self.indices))
        
        fire_count = 0
        no_fire_count = 0
        
        print(f"📊 Analizando distribución de clases en {n_samples} muestras...")
        
        for i in range(n_samples):
            _, y = self[i]
            if y.item() > 0.5:
                fire_count += 1
            else:
                no_fire_count += 1
        
        total = fire_count + no_fire_count
        fire_pct = (fire_count / total) * 100 if total > 0 else 0
        no_fire_pct = (no_fire_count / total) * 100 if total > 0 else 0
        
        print(f"   🔥 Con fuego: {fire_count} ({fire_pct:.1f}%)")
        print(f"   💧 Sin fuego: {no_fire_count} ({no_fire_pct:.1f}%)")
        
        if abs(fire_pct - 50) > 20:  # Si está muy desbalanceado
            print(f"   ⚠️ Dataset desbalanceado. Considera usar balance_classes=True")
        
        return {
            'fire_samples': fire_count,
            'no_fire_samples': no_fire_count,
            'fire_percentage': fire_pct,
            'balance_ratio': min(fire_count, no_fire_count) / max(fire_count, no_fire_count)
        }

    def get_sample_stats(self, n_samples=100):
        """Estadísticas generales del dataset"""
        n_samples = min(n_samples, len(self.indices))
        
        shapes = []
        fire_areas = []
        
        print(f"📊 Analizando {n_samples} muestras...")
        
        for i in range(n_samples):
            x, y = self[i]
            shapes.append(x.shape)
            
            # Para ignición, y es escalar, pero podemos obtener el área del mapa original
            sample_info = self.indices[i]
            t_next = sample_info["time_index"] + 1
            if t_next < len(self.datacube.time):
                y_map = self.datacube["is_fire"].isel(time=t_next).values
                fire_areas.append(np.sum(y_map))
        
        print(f"   📐 Shape de entrada: {shapes[0]}")
        print(f"   🔥 Área promedio de fuego: {np.mean(fire_areas):.1f} píxeles")
        print(f"   🔥 Área máxima de fuego: {np.max(fire_areas):.0f} píxeles")
        
        return {
            'input_shape': shapes[0],
            'avg_fire_area': np.mean(fire_areas),
            'max_fire_area': np.max(fire_areas)
        }


# Función para crear datasets de ignición con configuración completa
def create_ignition_datasets(datacube, splits, temporal_context=3, balance_train=True):
    """
    Crea datasets de ignición para train/val/test con configuración optimizada
    
    Args:
        datacube: xarray Dataset
        splits: dict con 'train', 'val', 'test' indices
        temporal_context: días de contexto temporal
        balance_train: si balancear clases en entrenamiento
    
    Returns:
        dict: datasets de ignición
    """
    print("🔥 CREANDO DATASETS DE IGNICIÓN...")
    
    # Dataset de entrenamiento (balanceado)
    ign_train = IgnitionDataset(
        datacube, splits['train'], 
        temporal_context=temporal_context,
        balance_classes=balance_train
    )
    
    # Datasets de validación y test (sin balancear para evaluación real)
    ign_val = IgnitionDataset(
        datacube, splits['val'],
        temporal_context=temporal_context,
        balance_classes=False
    )
    
    ign_test = IgnitionDataset(
        datacube, splits['test'],
        temporal_context=temporal_context,
        balance_classes=False
    )
    
    # Mostrar estadísticas
    print("\n📊 ESTADÍSTICAS DE IGNICIÓN:")
    print("TRAIN:")
    ign_train.get_class_distribution(n_samples=1000)
    print("\nVALIDATION:")
    ign_val.get_class_distribution(n_samples=500)
    
    return {
        'train': ign_train,
        'val': ign_val,
        'test': ign_test
    }

print("✅ IgnitionDataset mejorado implementado!")
print("🚀 Usa create_ignition_datasets(datacube, splits) para configurar todo")