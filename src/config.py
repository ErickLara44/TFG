from pathlib import Path
import os

from dotenv import load_dotenv
from loguru import logger


# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Folder paths
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUTS_DIR = PROJ_ROOT / "outputs"
AEMET_VALIDATION_RESUTLS_DIR = OUTPUTS_DIR / "AEMET_validation_results"
FIRERISK_MAPS_DIR = OUTPUTS_DIR / "FireRiskMaps"
VEGETATION_INDICES_FOLDER = os.getenv("VEGETATION_INDICES_FOLDER")

# Specific files paths
DATACUBE_PATH = os.getenv("DATACUBE_PATH")
VALIDATION_DATASETS_FOLDER = os.getenv("VALIDATION_DATASETS_FOLDER")
AEMET_STATIONS_FILE = os.getenv("AEMET_STATIONS_FILE")
RAILWAYS_GPKG_FILE = os.getenv("RAILWAYS_GPKG_FILE")
BASE_RASTER_FILE = os.getenv("BASE_RASTER_FILE")
AUTONOMOUS_COMMUNITIES_FILE = os.getenv("AUTONOMOUS_COMMUNITIES_FILE")

# API KEYS:
ERA5_API_KEY = os.getenv("ERA5_API_KEY")






# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# === RUTAS DE ARCHIVOS ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" 
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Crear directorios si no existen
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True)

# Rutas específicas
DATACUBE_PATH = '/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc'
STATS_PATH = 'iberfire_normalization_stats.pkl'
MODEL_SAVE_PATH = 'best_spanish_fire_model.pth'

# === PARÁMETROS DEL MODELO ===
MODEL_CONFIG = {
    'window_size': 64,
    'temporal_context': 3,
    'hidden_dim': 256,
    'spatial_features_dim': 512
}

# === PARÁMETROS DE ENTRENAMIENTO ===
TRAINING_CONFIG = {
    'batch_size': 17,
    'epochs': 10,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'device': 'mps',  # 'mps', 'cuda', o 'cpu'
    'num_workers': 0
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
DEFAULT_VARIABLES = {
    'elevation_mean': 'static', 
    'slope_mean': 'static', 
    'CLC_2018_forest_proportion': 'static', 
    'CLC_2018_scrub_proportion': 'static', 
    'CLC_2018_agricultural_proportion': 'static', 
    'dist_to_roads_mean': 'static', 
    'popdens_2018': 'static', 
    'is_waterbody': 'static', 
    't2m_mean': 'dynamic', 
    'RH_min': 'dynamic', 
    'wind_speed_mean': 'dynamic', 
    'total_precipitation_mean': 'dynamic', 
    'NDVI': 'dynamic', 
    'SWI_010': 'dynamic', 
    'FWI': 'dynamic', 
    'LST': 'dynamic'
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