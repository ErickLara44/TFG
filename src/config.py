from pathlib import Path
import os

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# === RUTAS DE ARCHIVOS ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # src/config.py -> src/ -> PROJECT_ROOT

# Folder paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Rutas específicas
DATACUBE_PATH = os.getenv("DATACUBE_PATH", str(DATA_DIR / "IberFire.nc"))
STATS_PATH = DATA_DIR / 'processed' / 'iberfire_normalization_stats.pkl'
MODEL_SAVE_PATH = PROJECT_ROOT / 'best_robust_ignition_model.pth'

# API KEYS
ERA5_API_KEY = os.getenv("ERA5_API_KEY")

# Configurar tqdm con loguru si está instalado
try:
    from tqdm import tqdm
    if hasattr(logger, "remove"):
        logger.remove(0)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, AttributeError):
    pass

# === PARÁMETROS DEL MODELO ===
MODEL_CONFIG = {
    'window_size': 64,
    'temporal_context': 3,
    'hidden_dim': 256,
    'spatial_features_dim': 512
}

# === PARÁMETROS DE ENTRENAMIENTO ===
TRAINING_CONFIG = {
    'batch_size': 4,
    'epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'device': 'mps',  # 'mps', 'cuda', o 'cpu'
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True
}

# === PARÁMETROS DEL DATASET ===
DATASET_CONFIG = {
    'start_year': 2020,
    'end_year': 2024,
    'window_size': MODEL_CONFIG['window_size'],
    'temporal_context': MODEL_CONFIG['temporal_context'],
    'normalize': True,
    'test_size': 0.2,
    'random_state': 42
}

# === VARIABLES DEL MODELO ===
# === VARIABLES DEL MODELO ===
DEFAULT_VARIABLES = {
    # Static / Topographic
    'elevation_mean': 'static', 
    'slope_mean': 'static', 
    'dist_to_roads_mean': 'static', 
    'popdens_2018': 'static', 
    
    # Dynamic Land Cover (Replacing static CLC_2018)
    'CLC_current_forest_proportion': 'dynamic',
    'CLC_current_scrub_proportion': 'dynamic',
    'CLC_current_agricultural_proportion': 'dynamic',
    
    # Fire Weather & Physics
    'hydric_stress': 'dynamic',      # LST - t2m
    'solar_risk': 'dynamic',         # Aspect * t2m_max
    'wind_u': 'dynamic',             # East-West Vector
    'wind_v': 'dynamic',             # North-South Vector
    'structural_drought': 'dynamic', # SWI Deep - Shallow
    
    # Raw Meteorologic (Keep basic ones if needed, but remove redundant)
    'total_precipitation_mean': 'dynamic', 
    'NDVI': 'dynamic', 
    'FWI': 'dynamic',
    
    # Anti-Leakage
    'is_near_fire_lag1': 'dynamic',  # T-1 neighbor status
    
    # Human Factors
    'is_weekend': 'dynamic',
    'is_waterbody': 'static',
}

# === CONFIGURACIÓN DE VISUALIZACIÓN ===
VIZ_CONFIG = {
    'figure_size': (12, 10),
    'dpi': 300,
    'show_plots': True,
    'save_plots': True
}

# === CONFIGURACIÓN DE LOGGING ===
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# === MENSAJES Y EMOJIS ===
MESSAGES = {
    'loading_data': "🔥 Cargando datos...",
    'training_start': "🚀 Iniciando entrenamiento...",
    'training_complete': "✅ Entrenamiento completado",
    'model_saved': "💾 Modelo guardado",
    'prediction_start': "🎯 Haciendo predicción...",
    'map_generated': "🗺️ Mapa generado"
}