import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# --- FUNCIONES DE OPTIMIZACIÓN DE ESTADÍSTICAS ---

def precompute_and_save_stats(datacube_path, output_path='normalization_stats.pkl', sample_size=1000):
    """
    Pre-calcular estadísticas y guardarlas en un archivo para uso posterior.
    Solo necesitas ejecutar esto UNA VEZ.
    """
    print("🔥 Pre-calculando estadísticas de normalización...")
    
    # Cargar datacube
    if datacube_path.endswith('.nc'):
        ds = xr.open_dataset(datacube_path)
    elif datacube_path.endswith('.zarr'):
        ds = xr.open_zarr(datacube_path)
    else:
        raise ValueError("Formato no soportado. Usar .nc o .zarr")
    
    # Variables por defecto
    selected_variables = {
        'elevation_mean': 'static', 'slope_mean': 'static', 
        'CLC_2018_forest_proportion': 'static', 'CLC_2018_scrub_proportion': 'static', 
        'CLC_2018_agricultural_proportion': 'static', 'dist_to_roads_mean': 'static', 
        'popdens_2018': 'static', 'is_waterbody': 'static', 't2m_mean': 'dynamic', 
        'RH_min': 'dynamic', 'wind_speed_mean': 'dynamic', 
        'total_precipitation_mean': 'dynamic', 'NDVI': 'dynamic', 
        'SWI_010': 'dynamic', 'FWI': 'dynamic', 'LST': 'dynamic',
        'is_near_fire': 'dynamic'
    }
    
    stats = {}
    
    # Dimensiones
    x_size = len(ds.x) if 'x' in ds.dims else 100
    y_size = len(ds.y) if 'y' in ds.dims else 100
    time_size = len(ds.time) if 'time' in ds.dims else 1
    
    for var_name, var_type in selected_variables.items():
        if var_name not in ds or var_name == 'is_fire':
            continue
            
        print(f"📊 Procesando {var_name}...")
        
        var_data = ds[var_name]
        
        # Muestreo inteligente
        if var_type == 'dynamic' and 'time' in var_data.dims:
            # Muestrear menos puntos temporales y espaciales
            t_sample = min(20, time_size)  # Máximo 20 tiempos
            x_sample = min(50, x_size)     # Máximo 50 puntos en X
            y_sample = min(50, y_size)     # Máximo 50 puntos en Y
            
            t_indices = np.random.choice(time_size, size=t_sample, replace=False)
            x_indices = np.random.choice(x_size, size=x_sample, replace=False)
            y_indices = np.random.choice(y_size, size=y_sample, replace=False)
            
            sampled_data = var_data.isel(time=t_indices, x=x_indices, y=y_indices)
        else:
            # Variables estáticas
            x_sample = min(100, x_size)
            y_sample = min(100, y_size)
            
            x_indices = np.random.choice(x_size, size=x_sample, replace=False)
            y_indices = np.random.choice(y_size, size=y_sample, replace=False)
            
            sampled_data = var_data.isel(x=x_indices, y=y_indices)
        
        # Calcular estadísticas
        data_values = sampled_data.values.flatten()
        data_values = data_values[~np.isnan(data_values)]
        
        if len(data_values) > 0:
            stats[var_name] = {
                'mean': float(np.mean(data_values)),
                'std': float(np.std(data_values))
            }
            print(f"   ✅ mean={stats[var_name]['mean']:.3f}, std={stats[var_name]['std']:.3f}")
        else:
            stats[var_name] = {'mean': 0.0, 'std': 1.0}
            print(f"   ⚠️ Sin datos válidos, usando valores por defecto")
    
    # Guardar estadísticas
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"💾 Estadísticas guardadas en {output_path}")
    print(f"✅ ¡Listo! Ahora usa stats_path='{output_path}' en tu dataset")
    return stats


def load_precomputed_stats(stats_path='normalization_stats.pkl'):
    """
    Cargar estadísticas pre-calculadas.
    """
    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        print(f"✅ Estadísticas cargadas desde {stats_path}")
        return stats
    except FileNotFoundError:
        print(f"❌ Archivo {stats_path} no encontrado")
        return None


# --- 1. Definición del Dataset OPTIMIZADA ---

class SpanishFireDataset(Dataset):
    """
    Dataset específico para predicción de fuego en España usando las variables seleccionadas.
    VERSIÓN OPTIMIZADA con cálculo rápido de estadísticas.
    """
    
    def __init__(self, datacube_path, fire_events_path=None, 
                 start_year=2020, end_year=2024,
                 window_size=64, temporal_context=3, 
                 selected_variables=None, normalize=True,
                 stats_path='normalization_stats.pkl',  # NUEVO PARÁMETRO
                 precompute_stats=False,  # NUEVO PARÁMETRO
                 valid_samples=None):  # Para re-instanciación
        
        self.window_size = window_size
        self.temporal_context = temporal_context
        self.normalize = normalize
        self.start_year = start_year
        self.end_year = end_year
        
        if selected_variables is None:
            selected_variables = self._get_default_variables()
        self.selected_variables = selected_variables
        
        print(f"🔥 Cargando datacube para España ({start_year}-{end_year})...")
        self.datacube = self._load_and_filter_datacube(datacube_path)
        
        print(f"📊 Variables seleccionadas: {len(self.selected_variables)}")
        for i, var in enumerate(self.selected_variables):
            print(f"   {i+1:2d}. {var}")
        
        self.use_datacube_labels = 'is_fire' in self.datacube.data_vars
        
        if self.use_datacube_labels:
            print("✅ Variable 'is_fire' encontrada en el datacube. Usándola como etiqueta.")
            self.fire_events = None
        elif fire_events_path:
            self.fire_events = self._load_fire_events(fire_events_path)
        else:
            print("⚠️ No se proporcionaron eventos de fuego. Generando sintéticos...")
            self.fire_events = self._generate_synthetic_fire_events()
        
        # Si se proporcionan muestras válidas (para re-instanciación), usarlas
        if valid_samples is not None:
            self.valid_samples = valid_samples
            print(f"✅ Usando {len(self.valid_samples)} muestras proporcionadas")
        else:
            print("🎯 Generando muestras válidas...")
            self.valid_samples = self._generate_valid_samples()
            print(f"✅ {len(self.valid_samples)} muestras válidas generadas")
        
        # OPTIMIZACIÓN DE NORMALIZACIÓN
        if self.normalize:
            print("📈 Configurando normalización...")
            
            if precompute_stats:
                # Opción 1: Pre-calcular y guardar (solo la primera vez)
                print("⚡ Pre-calculando estadísticas...")
                self.normalization_stats = precompute_and_save_stats(
                    datacube_path, stats_path, sample_size=1000
                )
            else:
                # Opción 2: Cargar estadísticas pre-calculadas (recomendado)
                self.normalization_stats = load_precomputed_stats(stats_path)
                
                if self.normalization_stats is None:
                    # Si no existen, calcular con muestreo rápido
                    print("⚡ Calculando estadísticas con muestreo rápido...")
                    self.normalization_stats = self._compute_normalization_stats_fast(sample_size=1000)
                    
                    # Guardar para próximas veces
                    with open(stats_path, 'wb') as f:
                        pickle.dump(self.normalization_stats, f)
                    print(f"💾 Estadísticas guardadas en {stats_path} para uso futuro")
    
    def _get_default_variables(self):
        return {
            'elevation_mean': 'static','slope_mean': 'static', 
            'CLC_2018_forest_proportion': 'static', 'CLC_2018_scrub_proportion': 'static', 
            'CLC_2018_agricultural_proportion': 'static', 'dist_to_roads_mean': 'static', 
            'popdens_2018': 'static', 'is_waterbody': 'static', 't2m_mean': 'dynamic', 
            'RH_min': 'dynamic', 'wind_speed_mean': 'dynamic', 
            'total_precipitation_mean': 'dynamic', 'NDVI': 'dynamic', 
            'SWI_010': 'dynamic', 'FWI': 'dynamic', 'LST': 'dynamic',
            'is_near_fire': 'dynamic'
        }
    
    def _load_and_filter_datacube(self, datacube_path):
        if datacube_path.endswith('.nc'):
            ds = xr.open_dataset(datacube_path)
        elif datacube_path.endswith('.zarr'):
            ds = xr.open_zarr(datacube_path)
        else:
            raise ValueError("Formato no soportado. Usar .nc o .zarr")
        
        start_date = f'{self.start_year}-01-01'
        end_date = f'{self.end_year}-12-31'
        
        if 'time' in ds.dims:
            ds_filtered = ds.sel(time=slice(start_date, end_date))
        else:
            ds_filtered = ds
        
        available_vars = list(ds_filtered.data_vars.keys()) + list(ds_filtered.coords.keys())
        missing_vars = [var for var in self.selected_variables.keys() if var not in available_vars and var != 'is_fire']
        
        if missing_vars:
            print(f"   ⚠️ Variables no encontradas: {missing_vars}")
            self.selected_variables = {k: v for k, v in self.selected_variables.items() if k not in missing_vars}
        
        if 'is_fire' in ds_filtered.data_vars:
            self.selected_variables['is_fire'] = 'dynamic'
        
        return ds_filtered
    
    def _load_fire_events(self, fire_events_path):
        if fire_events_path.endswith('.csv'):
            df = pd.read_csv(fire_events_path)
            df['date'] = pd.to_datetime(df['date'])
            df_filtered = df[(df['date'].dt.year >= self.start_year) & (df['date'].dt.year <= self.end_year)]
            return df_filtered
        else:
            return np.load(fire_events_path)
    
    def _generate_synthetic_fire_events(self):
        np.random.seed(42)
        start_date = pd.to_datetime(f'{self.start_year}-01-01')
        end_date = pd.to_datetime(f'{self.end_year}-12-31')
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        events = []
        x_size = len(self.datacube.x) if 'x' in self.datacube.dims else 1000
        y_size = len(self.datacube.y) if 'y' in self.datacube.dims else 1000
        
        for _ in range(2000):
            selected_date = np.random.choice(date_range)
            events.append({'date': selected_date, 'x_idx': np.random.randint(0, x_size), 'y_idx': np.random.randint(0, y_size)})
        
        return pd.DataFrame(events)

    def _generate_valid_samples(self):
        valid_samples = []
        if 'time' in self.datacube.dims:
            time_coords = self.datacube.time.values
            time_indices = range(self.temporal_context, len(time_coords))
        else:
            time_indices = [0]
        
        x_size = len(self.datacube.x) if 'x' in self.datacube.dims else 1000
        y_size = len(self.datacube.y) if 'y' in self.datacube.dims else 1000
        x_step = max(1, self.window_size // 4)
        y_step = max(1, self.window_size // 4)
        
        sample_count = 0
        for t_idx in time_indices:
            current_date = pd.to_datetime(self.datacube.time.values[t_idx])
            for x_start in range(0, x_size - self.window_size, x_step):
                for y_start in range(0, y_size - self.window_size, y_step):
                    fire_label = self._get_fire_label(current_date, x_start, y_start)
                    valid_samples.append({'t_idx': t_idx, 'x_start': x_start, 'y_start': y_start, 'date': current_date, 'fire_label': fire_label})
                    sample_count += 1
                    if sample_count >= 10000: break
                if sample_count >= 10000: break
            if sample_count >= 10000: break
        return valid_samples
    
    def _get_fire_label(self, date, x_start, y_start):
        if self.use_datacube_labels:
            time_coords = self.datacube.time.values
            t_idx = np.where(time_coords == np.datetime64(date))[0][0]
            fire_label_data = self.datacube['is_fire'].isel(
                time=t_idx,
                x=slice(x_start, x_start + self.window_size),
                y=slice(y_start, y_start + self.window_size)
            )
            fire_occurred = (fire_label_data.values > 0).any()
            return 1 if fire_occurred else 0
        else:
            if isinstance(self.fire_events, pd.DataFrame):
                nearby_events = self.fire_events[
                    (self.fire_events['date'] == date) &
                    (self.fire_events['x_idx'] >= x_start) & (self.fire_events['x_idx'] < x_start + self.window_size) &
                    (self.fire_events['y_idx'] >= y_start) & (self.fire_events['y_idx'] < y_start + self.window_size)
                ]
                return 1 if len(nearby_events) > 0 else 0
            else:
                return 0
    
    def _compute_normalization_stats_fast(self, sample_size=1000):
        """
        Versión ultra-rápida del cálculo de estadísticas con muestreo agresivo.
        """
        stats = {}
        print(f"⚡ Cálculo rápido de estadísticas (n={sample_size})...")
        
        # Obtener dimensiones
        x_size = len(self.datacube.x) if 'x' in self.datacube.dims else 100
        y_size = len(self.datacube.y) if 'y' in self.datacube.dims else 100
        time_size = len(self.datacube.time) if 'time' in self.datacube.dims else 1
        
        for var_name, var_type in self.selected_variables.items():
            if var_name not in self.datacube or var_name == 'is_fire':
                continue
                
            print(f"   {var_name}...", end='')
            
            try:
                var_data = self.datacube[var_name]
                
                # Muestreo muy agresivo para velocidad máxima
                if var_type == 'dynamic' and 'time' in var_data.dims:
                    # Solo 5 tiempos aleatorios y 10x10 píxeles
                    t_idx = np.random.choice(time_size, size=min(5, time_size), replace=False)
                    x_idx = np.random.choice(x_size, size=min(10, x_size), replace=False)
                    y_idx = np.random.choice(y_size, size=min(10, y_size), replace=False)
                    
                    sample = var_data.isel(time=t_idx, x=x_idx, y=y_idx).values.flatten()
                else:
                    # Variables estáticas: 20x20 píxeles
                    x_idx = np.random.choice(x_size, size=min(20, x_size), replace=False)
                    y_idx = np.random.choice(y_size, size=min(20, y_size), replace=False)
                    
                    sample = var_data.isel(x=x_idx, y=y_idx).values.flatten()
                
                # Limpiar y calcular
                sample = sample[~np.isnan(sample)]
                
                if len(sample) > 0:
                    stats[var_name] = {
                        'mean': float(np.mean(sample)),
                        'std': float(np.std(sample)) if np.std(sample) > 0 else 1.0
                    }
                    print(f" ✓ ({stats[var_name]['mean']:.2f}±{stats[var_name]['std']:.2f})")
                else:
                    stats[var_name] = {'mean': 0.0, 'std': 1.0}
                    print(" ⚠️ (default)")
                    
            except Exception as e:
                stats[var_name] = {'mean': 0.0, 'std': 1.0}
                print(f" ❌ (error: {str(e)[:30]}...)")
        
        return stats
    
    def _extract_sample_data(self, sample_info):
        t_idx = sample_info['t_idx']
        x_start, y_start = sample_info['x_start'], sample_info['y_start']
        x_end, y_end = x_start + self.window_size, y_start + self.window_size
        start_t, end_t = t_idx - self.temporal_context, t_idx
        
        time_series_data = self.datacube.isel(time=slice(start_t, end_t), x=slice(x_start, x_end), y=slice(y_start, y_end))
        
        sequence_tensors = []
        for t in range(self.temporal_context):
            daily_channels = []
            for var_name, var_type in self.selected_variables.items():
                if var_name == 'is_fire': continue
                var_array = time_series_data[var_name].isel(time=t).values if var_type == 'dynamic' else time_series_data[var_name].values
                if self.normalize and var_name in self.normalization_stats:
                    stats = self.normalization_stats[var_name]
                    if stats['std'] > 0: var_array = (var_array - stats['mean']) / stats['std']
                daily_channels.append(var_array)
            daily_tensor = np.stack(daily_channels, axis=0)
            sequence_tensors.append(daily_tensor)
        
        return np.stack(sequence_tensors, axis=0)
    
    def get_variable_names(self):
        """Método para obtener los nombres de variables (necesario para el modelo)"""
        return [var for var in self.selected_variables.keys() if var != 'is_fire']
    
    def __len__(self): return len(self.valid_samples)
    def __getitem__(self, idx):
        sample_info = self.valid_samples[idx]
        data = self._extract_sample_data(sample_info)
        label = sample_info['fire_label']
        return torch.FloatTensor(data), torch.FloatTensor([label])

# --- 2. Definición del Modelo ---

class SpanishFirePredictionModel(nn.Module):
    def __init__(self, num_input_channels, temporal_context, spatial_features_dim=512, hidden_dim=256):
        super(SpanishFirePredictionModel, self).__init__()
        self.temporal_context = temporal_context
        
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor.fc = nn.Identity()
        
        self.convlstm = ResNetConvLSTM(
            input_channels=num_input_channels,
            hidden_channels=[hidden_dim],
            kernel_size=(3, 3),
            num_layers=1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 17, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        convlstm_output = self.convlstm(x)[0][-1]
        
        convlstm_output_flattened = convlstm_output.view(batch_size, -1)
        
        out = self.classifier(convlstm_output_flattened)
        return out.squeeze(1)

# --- 3. Definición del ConvLSTM ---

class ResNetConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ResNetConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_channels,
                                          hidden_dim=self.hidden_channels[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
            
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = (kernel_size, kernel_size)
        return kernel_size

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        h, w = image_size
        return (torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv.weight.device))

# --- 4. Funciones de entrenamiento y evaluación ---

def train_spanish_fire_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='mps'):
    """
    Entrenar modelo específicamente para datos españoles con barra de progreso.
    """
    
    # Lógica de selección de dispositivo, ahora compatible con Apple Silicon (MPS)
    if device == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Usando Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Usando NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU no disponible, usando CPU")
    
    model = model.to(device)
    print(f"✅ Modelo movido a {device}")
    
    fire_count = sum([label.item() for _, label in train_loader.dataset])
    total_count = len(train_loader.dataset)
    pos_weight = torch.tensor([(total_count - fire_count) / fire_count]).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLOnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=True)
    
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')
    
    print(f"📈 Peso de clase positiva (fuego): {pos_weight.item():.2f}")
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Envuelve el dataloader con tqdm para la barra de progreso
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)')
        
        for data, target in train_loader_tqdm:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            predicted = (output > 0.5).float()
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validación
        model.eval()
        val_loss = 0.0
        val_true_positives, val_false_positives, val_false_negatives = 0, 0, 0
        
        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Val)', leave=False)

        with torch.no_grad():
            for data, target in val_loader_tqdm:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                predicted = (output > 0.5).float()
                
                val_true_positives += ((predicted == 1) & (target == 1)).sum().item()
                val_false_positives += ((predicted == 1) & (target == 0)).sum().item()
                val_false_negatives += ((predicted == 0) & (target == 1)).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        precision = val_true_positives / (val_true_positives + val_false_positives) if (val_true_positives + val_false_positives) > 0 else 0
        recall = val_true_positives / (val_true_positives + val_false_negatives) if (val_true_positives + val_false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
        print('-' * 50)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict(), 'f1_score': f1}, 'best_spanish_fire_model.pth')
            print(f'✅ Mejor modelo guardado (F1: {f1:.3f})')
    
    return {'train_losses': train_losses, 'val_losses': val_losses}


# --- 5. FUNCIONES DE PREDICCIÓN Y MAPEO INTEGRADAS ---

def predict_and_map(model, datacube, sample_data, sample_coordinates, device='cpu'):
    """
    Función integrada: hace predicción Y muestra mapa si hay riesgo de fuego
    
    Args:
        model: Tu modelo entrenado
        datacube: Tu datacube con los mapas de España
        sample_data: Datos de entrada para la predicción [batch, time, channels, H, W]
        sample_coordinates: Coordenadas de la muestra (x_start, y_start, t_idx)
        device: Dispositivo de cálculo
    
    Returns:
        prediction: Probabilidad de fuego (0-1)
        fig: Figura del mapa (si hay riesgo alto)
    """
    
    print("🔥 Haciendo predicción...")
    
    # 1. PREDICCIÓN con tu modelo
    model.eval()
    model = model.to(device)
    sample_data = sample_data.to(device)
    
    with torch.no_grad():
        prediction = model(sample_data).cpu().item()
    
    print(f"🎯 Probabilidad de fuego: {prediction:.1%}")
    
    # 2. INTERPRETACIÓN del resultado
    if prediction > 0.8:
        riesgo_nivel = "🚨 CRÍTICO"
        color_alerta = "red"
    elif prediction > 0.6:
        riesgo_nivel = "⚠️ ALTO"  
        color_alerta = "orange"
    elif prediction > 0.4:
        riesgo_nivel = "🟡 MODERADO"
        color_alerta = "yellow"
    else:
        riesgo_nivel = "✅ BAJO"
        color_alerta = "green"
    
    print(f"📊 Nivel de riesgo: {riesgo_nivel}")
    
    # 3. MOSTRAR MAPA si hay riesgo significativo
    fig = None
    if prediction > 0.5:  # Solo mostrar mapa si hay riesgo
        print("🗺️ Generando mapa del área afectada...")
        fig = mostrar_mapa_area_afectada(
            datacube, sample_coordinates, prediction, riesgo_nivel
        )
    else:
        print("✅ Riesgo bajo - No se genera mapa")
    
    return prediction, fig

def mostrar_mapa_area_afectada(datacube, coordinates, prediction, riesgo_nivel):
    """
    Mostrar mapa de España con el área específica donde se predice el fuego
    
    Args:
        datacube: Tu datacube IberFire con mapas
        coordinates: (x_start, y_start, t_idx) de la predicción
        prediction: Probabilidad predicha
        riesgo_nivel: Texto del nivel de riesgo
    """
    
    x_start, y_start, t_idx = coordinates
    
    # Obtener mapa base de España del datacube
    print("🗺️ Extrayendo mapa base de España...")
    
    # Usar una variable como base del mapa (ej: elevación)
    if 'elevation_mean' in datacube.data_vars:
        mapa_base = datacube['elevation_mean'].values
    elif 'is_fire' in datacube.data_vars:
        # O usar el mapa de fuego histórico si existe
        mapa_base = datacube['is_fire'].isel(time=t_idx).values
    else:
        # Crear mapa básico con las dimensiones del datacube
        mapa_base = np.ones((len(datacube.y), len(datacube.x)))
    
    # Obtener coordenadas geográficas
    if 'lat' in datacube.coords and 'lon' in datacube.coords:
        lats = datacube.lat.values
        lons = datacube.lon.values
        lat_coord_name, lon_coord_name = 'lat', 'lon'
    else:
        # Usar índices como coordenadas aproximadas
        lats = datacube.y.values
        lons = datacube.x.values
        lat_coord_name, lon_coord_name = 'y', 'x'
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. MAPA BASE de España
    im_base = ax.imshow(mapa_base, extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                       cmap='terrain', alpha=0.6, aspect='auto', origin='lower')
    
    # 2. MARCAR ÁREA DE PREDICCIÓN
    # Calcular coordenadas reales del área predicha
    window_size = 64  # Tu ventana de predicción
    
    if x_start + window_size < len(lons) and y_start + window_size < len(lats):
        # Coordenadas del área afectada
        lat_min = lats[y_start]
        lat_max = lats[y_start + window_size]
        lon_min = lons[x_start]  
        lon_max = lons[x_start + window_size]
        
        # Color según el nivel de riesgo
        if prediction > 0.8:
            color_zona = 'red'
            alpha_zona = 0.8
        elif prediction > 0.6:
            color_zona = 'orange'
            alpha_zona = 0.7
        else:
            color_zona = 'yellow'
            alpha_zona = 0.6
        
        # Crear rectángulo del área afectada
        area_rect = Rectangle(
            (lon_min, lat_min), 
            lon_max - lon_min, 
            lat_max - lat_min,
            linewidth=3, 
            edgecolor=color_zona, 
            facecolor=color_zona,
            alpha=alpha_zona,
            label=f'Área de riesgo ({prediction:.1%})'
        )
        ax.add_patch(area_rect)
        
        # Marcar centro del área
        centro_lat = (lat_min + lat_max) / 2
        centro_lon = (lon_min + lon_max) / 2
        ax.plot(centro_lon, centro_lat, 'o', color='darkred', markersize=10, 
                markeredgecolor='white', markeredgewidth=2)
        ax.text(centro_lon, centro_lat + 0.1, f'{prediction:.1%}', 
                ha='center', va='bottom', fontweight='bold', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # 3. CONFIGURACIÓN DEL MAPA
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Título dinámico
    fecha_actual = datacube.time.values[t_idx] if 'time' in datacube.coords else "Hoy"
    if hasattr(fecha_actual, 'strftime'):
        fecha_str = pd.to_datetime(fecha_actual).strftime('%d/%m/%Y')
    else:
        fecha_str = str(fecha_actual)[:10]
    
    plt.title(f'🔥 PREDICCIÓN DE INCENDIO - ESPAÑA\n'
             f'{riesgo_nivel} | Probabilidad: {prediction:.1%}\n'
             f'📅 {fecha_str} | 📍 Coordenadas: ({y_start}, {x_start})',
             fontsize=14, fontweight='bold', pad=20)
    
    # 4. INFORMACIÓN ADICIONAL
    info_text = f"""📊 INFORMACIÓN DE LA PREDICCIÓN:
🎯 Probabilidad: {prediction:.1%}
⚠️ Nivel: {riesgo_nivel.split()[1] if len(riesgo_nivel.split()) > 1 else riesgo_nivel}
📐 Área analizada: 64x64 píxeles
📍 Coordenadas: x={x_start}, y={y_start}"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
    
    # Leyenda
    if prediction > 0.5:
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig

def test_prediction_with_mapping(modelo_path, datacube_path):
    """
    Función para probar el sistema de predicción + mapeo con una muestra aleatoria
    """
    print("🔥 SISTEMA DE PREDICCIÓN INTEGRADO CON MAPEO")
    print("="*60)
    
    # Cargar modelo
    try:
        checkpoint = torch.load(modelo_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Recrear modelo con parámetros correctos
            model = SpanishFirePredictionModel(
                num_input_channels=17,  # Ajustar según tus variables
                temporal_context=3
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Modelo cargado desde checkpoint (F1: {checkpoint.get('f1_score', 'N/A'):.3f})")
        else:
            model = checkpoint
            print("✅ Modelo cargado directamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None
    
    # Crear dataset para obtener una muestra
    try:
        dataset = SpanishFireDataset(
            datacube_path=datacube_path,
            stats_path='iberfire_normalization_stats.pkl',
            precompute_stats=False
        )
        print("✅ Dataset cargado")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return None
    
    # Tomar una muestra aleatoria
    sample_idx = np.random.randint(0, len(dataset))
    sample_data, sample_label = dataset[sample_idx]
    sample_info = dataset.valid_samples[sample_idx]
    
    print(f"🎲 Muestra seleccionada: #{sample_idx}")
    print(f"📍 Coordenadas: x={sample_info['x_start']}, y={sample_info['y_start']}")
    print(f"🏷️ Etiqueta real: {'🔥 SÍ HAY FUEGO' if sample_label.item() > 0.5 else '✅ NO HAY FUEGO'}")
    
    # PREDICCIÓN + MAPA
    coordinates = (sample_info['x_start'], sample_info['y_start'], sample_info['t_idx'])
    prediction, fig = predict_and_map(
        model=model,
        datacube=dataset.datacube, 
        sample_data=sample_data.unsqueeze(0),  # Agregar dimensión de batch
        sample_coordinates=coordinates,
        device='cpu'  # Cambiar según tu dispositivo
    )
    
    # Comparar con realidad
    print(f"\n📊 COMPARACIÓN:")
    print(f"🤖 Predicción del modelo: {prediction:.1%}")
    print(f"🏷️ Realidad: {'Fuego' if sample_label.item() > 0.5 else 'No fuego'}")
    
    acierto = (prediction > 0.5) == (sample_label.item() > 0.5)
    print(f"✅ ¿Acertó?: {'SÍ' if acierto else 'NO'}")
    
    if fig:
        plt.show()
        print("🗺️ Mapa mostrado - ¡Revisa la ventana gráfica!")
    
    return prediction, fig


# --- 6. Función auxiliar para crear datasets con muestras específicas ---

def create_dataset_with_samples(datacube_path, valid_samples, **kwargs):
    """
    Crear dataset con muestras específicas (para train/val split)
    """
    return SpanishFireDataset(
        datacube_path=datacube_path,
        valid_samples=valid_samples,
        **kwargs
    )

# --- 7. Bloque Principal de Ejecución OPTIMIZADO CON MAPEO INTEGRADO ---

if __name__ == "__main__":
    
    # 1. Configuración de rutas y parámetros
    datacube_path = '/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc' 
    stats_path = 'iberfire_normalization_stats.pkl'  # NUEVO: Archivo de estadísticas
    
    window_size = 64
    temporal_context = 3
    batch_size = 17
    epochs = 10
    lr = 0.001
    
    # 2. OPCIÓN A: Pre-calcular estadísticas (SOLO LA PRIMERA VEZ)
    print("="*60)
    print("🔥 CONFIGURACIÓN DE ESTADÍSTICAS")
    print("="*60)
    
    if not os.path.exists(stats_path):
        print(f"📊 Archivo de estadísticas no encontrado: {stats_path}")
        print("⚡ Pre-calculando estadísticas (esto tomará ~2-3 minutos)...")
        precompute_and_save_stats(
            datacube_path=datacube_path,
            output_path=stats_path,
            sample_size=1000
        )
        print("✅ ¡Estadísticas pre-calculadas y guardadas!")
    else:
        print(f"✅ Usando estadísticas existentes: {stats_path}")
    
    print("\n" + "="*60)
    print("🔥 CARGANDO DATASET")
    print("="*60)
    
    # 3. Cargar el dataset principal (con estadísticas optimizadas)
    full_dataset = SpanishFireDataset(
        datacube_path=datacube_path,
        window_size=window_size,
        temporal_context=temporal_context,
        normalize=True,
        stats_path=stats_path,  # Usar estadísticas pre-calculadas
        precompute_stats=False  # No recalcular
    )

    print("\n" + "="*60)
    print("🎯 DIVIDIENDO DATASET")
    print("="*60)
    
    # 4. Dividir el dataset
    train_samples, val_samples = train_test_split(
        full_dataset.valid_samples, 
        test_size=0.2, 
        random_state=42,
        stratify=[sample['fire_label'] for sample in full_dataset.valid_samples]
    )
    
    print(f"📊 Muestras de entrenamiento: {len(train_samples)}")
    print(f"📊 Muestras de validación: {len(val_samples)}")
    
    # Contar distribución de clases
    train_fire_count = sum(1 for sample in train_samples if sample['fire_label'] == 1)
    val_fire_count = sum(1 for sample in val_samples if sample['fire_label'] == 1)
    
    print(f"🔥 Fuegos en entrenamiento: {train_fire_count} ({100*train_fire_count/len(train_samples):.1f}%)")
    print(f"🔥 Fuegos en validación: {val_fire_count} ({100*val_fire_count/len(val_samples):.1f}%)")
    
    # 5. Crear datasets separados con las estadísticas ya calculadas
    print("\n⚡ Creando datasets de entrenamiento y validación...")
    
    train_dataset = SpanishFireDataset(
        datacube_path=datacube_path,
        window_size=window_size,
        temporal_context=temporal_context,
        normalize=True,
        stats_path=stats_path,
        precompute_stats=False,
        valid_samples=train_samples
    )
    
    val_dataset = SpanishFireDataset(
        datacube_path=datacube_path,
        window_size=window_size,
        temporal_context=temporal_context,
        normalize=True,
        stats_path=stats_path,
        precompute_stats=False,
        valid_samples=val_samples
    )
    
    print("\n" + "="*60)
    print("🚀 PREPARANDO ENTRENAMIENTO")
    print("="*60)
    
    # 6. Crear los DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0  # Cambiar a 2-4 si tienes problemas de memoria
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=0
    )
    
    # 7. Inicializar el modelo
    num_input_channels = len(full_dataset.get_variable_names())
    print(f"📊 Canales de entrada: {num_input_channels}")
    print(f"📊 Variables: {full_dataset.get_variable_names()}")
    
    model = SpanishFirePredictionModel(
        num_input_channels=num_input_channels,
        temporal_context=temporal_context,
        hidden_dim=256  # Puedes ajustar esto
    )
    
    # Contar parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Parámetros totales: {total_params:,}")
    print(f"🧠 Parámetros entrenables: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    # 8. Entrenar el modelo
    try:
        training_history = train_spanish_fire_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device='mps'  # Cambia a 'cuda' o 'cpu' según tu hardware
        )
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print("📁 Modelo guardado como: best_spanish_fire_model.pth")
        print(f"📊 Estadísticas guardadas en: {stats_path}")
        
        # Opcional: Plot de pérdidas
        try:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(training_history['train_losses'], label='Train Loss')
            plt.plot(training_history['val_losses'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
            print("📈 Gráfico de entrenamiento guardado: training_history.png")
            
        except ImportError:
            print("📈 Matplotlib no disponible, sin gráfico de entrenamiento")
        
        # 🆕 NUEVA FUNCIONALIDAD: Probar predicción con mapeo
        print("\n" + "="*60)
        print("🗺️ PROBANDO SISTEMA DE PREDICCIÓN + MAPEO")
        print("="*60)
        
        try:
            test_prediction_with_mapping(
                modelo_path='best_spanish_fire_model.pth',
                datacube_path=datacube_path
            )
        except Exception as e:
            print(f"⚠️ Error en el test de mapeo: {e}")
            print("💡 El modelo se entrenó correctamente, pero hay un problema con el mapeo")
            
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
        print("💾 Progreso guardado hasta el momento")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        print("💡 Verifica:")
        print("   - Que el archivo datacube existe y es accesible")
        print("   - Que tienes suficiente memoria disponible")
        print("   - Que las dimensiones del modelo son correctas")
        
    print("\n🎉 ¡Proceso completado!")
    print("\n" + "="*60)
    print("📋 FUNCIONES DISPONIBLES PARA USO:")
    print("="*60)
    print("🔥 predict_and_map(model, datacube, sample_data, coordinates)")
    print("   └─ Hace predicción + muestra mapa automáticamente")
    print()
    print("🗺️ test_prediction_with_mapping(modelo_path, datacube_path)")
    print("   └─ Prueba el sistema completo con muestra aleatoria")
    print()
    print("✅ ¡Tu modelo ya puede predecir Y mostrar mapas de España!")

# --- 8. FUNCIONES ADICIONALES PARA USO POSTERIOR ---

def load_trained_model(model_path):
    """Cargar modelo entrenado para uso posterior"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = SpanishFirePredictionModel(
                num_input_channels=17,
                temporal_context=3
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, checkpoint.get('f1_score', None)
        else:
            return checkpoint, None
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None

def quick_prediction_demo():
    """Demo rápido del sistema de predicción + mapeo"""
    print("🚀 DEMO RÁPIDO - PREDICCIÓN + MAPEO")
    print("="*40)
    
    model_path = 'best_spanish_fire_model.pth'
    datacube_path = '/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc'
    
    if os.path.exists(model_path):
        prediction, fig = test_prediction_with_mapping(model_path, datacube_path)
        print(f"✅ Demo completado - Predicción: {prediction:.1%}")
        return True
    else:
        print("❌ Modelo no encontrado. Entrena primero el modelo.")
        return False