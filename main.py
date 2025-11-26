# main.py
import torch
from torch.utils.data import DataLoader
import xarray as xr

DEFAULT_FEATURE_VARS = [
    'elevation_mean', 'slope_mean',
    'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion', 'CLC_2018_agricultural_proportion',
    'dist_to_roads_mean', 'popdens_2018', 'is_waterbody','CLC_2018_urban_fabric_proportion' , 
'CLC_2018_industrial_proportion',
'CLC_2018_artificial_proportion'
    't2m_mean', 'RH_min', 'wind_speed_mean', 'total_precipitation_mean',
    'NDVI', 'SWI_010', 'FWI', 'LST', 'is_near_fire','is_holiday'
]

# ========================================
# IMPORTACIONES CORREGIDAS
# ========================================

# --- Tabular ---
try:
    from data.data_tab import SpainCubeFireDataset
    from models.XGBoost import (
        SpainXGBoostPredictor,
        SpainLightGBMPredictor,
        SpainCatBoostPredictor,
        SpainRandomForestPredictor,
        calculate_fire_metrics,
        BaseFirePredictor
    )
    TABULAR_AVAILABLE = True
except ImportError:
    print("⚠️ Módulos tabulares no encontrados. Opciones 1–5 deshabilitadas.")
    TABULAR_AVAILABLE = False

# --- CNN Ignición ---
from data.data_ignition_improved import IgnitionDataset, create_ignition_datasets
from models.ignition import RobustFireIgnitionModel, train_robust_ignition_model

# --- CNN Propagación ---
from data.data_prop_improved import SpreadDataset, create_train_val_test_split
from models.prop import RobustFireSpreadModel, train_robust_spread_model


# ========================================
# FUNCIONES AUXILIARES
# ========================================

def setup_cnn_datasets(datacube, temporal_context_ign=7, temporal_context_spr=3):
    """Configura datasets robustos para ignición y propagación"""
    print("🔥 Configurando datasets CNN...")
    
    splits = create_train_val_test_split(
        datacube, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        min_temporal_context=max(temporal_context_ign, temporal_context_spr)
    )
    
    print("\n📊 Creando datasets de ignición...")
    ignition_datasets = create_ignition_datasets(
        datacube, splits, temporal_context=temporal_context_ign, balance_train=True
    )
    
    print("\n📊 Creando datasets de propagación...")
    spread_datasets = {
        'train': SpreadDataset(datacube, splits['train'], temporal_context=temporal_context_spr,
                               include_fire_state=True, filter_fire_samples=True),
        'val': SpreadDataset(datacube, splits['val'], temporal_context=temporal_context_spr,
                             include_fire_state=True, filter_fire_samples=True),
        'test': SpreadDataset(datacube, splits['test'], temporal_context=temporal_context_spr,
                              include_fire_state=True, filter_fire_samples=True)
    }
    
    return ignition_datasets, spread_datasets


def create_data_loaders(datasets, batch_sizes={'ignition': 32, 'spread': 16}):
    """Crea DataLoaders optimizados"""
    loaders = {}
    
    loaders['ignition'] = {
        'train': DataLoader(datasets['ignition']['train'], batch_size=batch_sizes['ignition'], shuffle=True),
        'val': DataLoader(datasets['ignition']['val'], batch_size=batch_sizes['ignition'], shuffle=False),
        'test': DataLoader(datasets['ignition']['test'], batch_size=batch_sizes['ignition'], shuffle=False)
    }
    
    loaders['spread'] = {
        'train': DataLoader(datasets['spread']['train'], batch_size=batch_sizes['spread'], shuffle=True),
        'val': DataLoader(datasets['spread']['val'], batch_size=batch_sizes['spread'], shuffle=False),
        'test': DataLoader(datasets['spread']['test'], batch_size=batch_sizes['spread'], shuffle=False)
    }
    
    return loaders


# ========================================
# MAIN
# ========================================

def main():
    datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/iberfire.nc"  # ⚠️ Cambia según tu ruta

    dataset_tabular = None
    train_X = val_X = test_X = None
    train_y = val_y = test_y = None
    datacube = None
    cnn_datasets = None
    data_loaders = None

    while True:
        print("\n" + "="*50)
        print("🔥 IBERFIRE - MENÚ PRINCIPAL")
        print("="*50)
        
        if TABULAR_AVAILABLE:
            print("📊 MODELOS TABULARES:")
            print("1: Preparar dataset tabular")
            print("2: Entrenar XGBoost")
            print("3: Entrenar LightGBM")
            print("4: Entrenar CatBoost")
            print("5: Entrenar Random Forest\n")
        
        print("🧠 MODELOS CNN ROBUSTOS:")
        print("6: Preparar datasets CNN (Ignición + Propagación)")
        print("7: Entrenar modelo robusto de IGNICIÓN")
        print("8: Entrenar modelo robusto de PROPAGACIÓN")
        print("9: Entrenar PIPELINE COMPLETO (Ignición → Propagación)")
        print("\n0: Salir")
        print("-"*50)
        
        choice = input("Selecciona una opción: ").strip()

        # ========================================
        # OPCIÓN 1: Preparar dataset tabular
        # ========================================
        if choice == '1' and TABULAR_AVAILABLE:
            try:
                dataset_tabular = SpainCubeFireDataset(datacube_path,
                 output_dir="data/processed",
                 start_year=2017, end_year=2024,
                 ccaa_target=None,
                 chunk_size_days=1,
                 neg_pos_ratio=1.5)
                
                print("🚀 Extrayendo datos...")
                dataset_tabular.extract_raw_data()
                
                print("⚖️ Balanceando y dividiendo...")
                dataset_tabular.balance_and_split()
                
                print("📏 Normalizando...")
                dataset_tabular.normalize_data()
                
                print("📦 Cargando datos preparados...")
                train_X, train_y, _ = dataset_tabular.get_features_labels("train")
                val_X, val_y, _ = dataset_tabular.get_features_labels("val")
                test_X, test_y, _ = dataset_tabular.get_features_labels("test")
                
                print(f"✅ Dataset tabular listo: Train={len(train_y)}, Val={len(val_y)}, Test={len(test_y)}")
            
            except Exception as e:
                print(f"❌ Error preparando dataset tabular: {e}")

        # ========================================
        # OPCIÓN 2: Entrenar XGBoost
        # ========================================
        elif choice == '2' and TABULAR_AVAILABLE:
            if train_X is None:
                print("❌ Primero prepara el dataset tabular (opción 1)")
                continue
            try:
                model = SpainXGBoostPredictor()
                model.train_with_validation_strategy(train_X, train_y, val_X, val_y)
                test_probs = model.predict(test_X)
                test_bin = (test_probs >= 0.5).astype(int)
                print("📊 Test XGBoost:", calculate_fire_metrics(test_y, test_bin, test_probs))
            except Exception as e:
                print(f"❌ Error entrenando XGBoost: {e}")

        # ========================================
        # OPCIONES 3–5: LightGBM, CatBoost, RF
        # ========================================
        elif choice == '3' and TABULAR_AVAILABLE:
            if train_X is None:
                print("❌ Primero prepara el dataset tabular (opción 1)")
                continue
            try:
                model = SpainLightGBMPredictor()
                model.train_with_validation_strategy(train_X, train_y, val_X, val_y)
                test_probs = model.predict(test_X)
                test_bin = (test_probs >= 0.5).astype(int)
                print("📊 Test LightGBM:", calculate_fire_metrics(test_y, test_bin, test_probs))
            except Exception as e:
                print(f"❌ Error entrenando LightGBM: {e}")

        elif choice == '4' and TABULAR_AVAILABLE:
            if train_X is None:
                print("❌ Primero prepara el dataset tabular (opción 1)")
                continue
            try:
                model = SpainCatBoostPredictor()
                model.train_with_validation_strategy(train_X, train_y, val_X, val_y)
                test_probs = model.predict(test_X)
                test_bin = (test_probs >= 0.5).astype(int)
                print("📊 Test CatBoost:", calculate_fire_metrics(test_y, test_bin, test_probs))
            except Exception as e:
                print(f"❌ Error entrenando CatBoost: {e}")

        elif choice == '5' and TABULAR_AVAILABLE:
            if train_X is None:
                print("❌ Primero prepara el dataset tabular (opción 1)")
                continue
            try:
                model = SpainRandomForestPredictor()
                model.train_with_validation_strategy(train_X, train_y, val_X, val_y)
                test_probs = model.predict(test_X)
                test_bin = (test_probs >= 0.5).astype(int)
                print("📊 Test RandomForest:", calculate_fire_metrics(test_y, test_bin, test_probs))
            except Exception as e:
                print(f"❌ Error entrenando RandomForest: {e}")

        # ========================================
        # OPCIÓN 6: Preparar datasets CNN
        # ========================================
        elif choice == '6':
            try:
                print("🔄 Cargando datacube...")
                datacube = xr.open_dataset(datacube_path)
                print(f"📊 Datacube cargado: {datacube.sizes}")
                
                print("🔄 Configurando datasets CNN robustos...")
                ignition_datasets, spread_datasets = setup_cnn_datasets(datacube)
                
                cnn_datasets = {
                    'ignition': ignition_datasets,
                    'spread': spread_datasets
                }
                
                print("🔄 Creando DataLoaders...")
                data_loaders = create_data_loaders(cnn_datasets)
                
                print("✅ Datasets CNN robustos listos!")
                print(f"   🔥 Ignición - Train: {len(cnn_datasets['ignition']['train'])}")
                print(f"   🔥 Propagación - Train: {len(cnn_datasets['spread']['train'])}")
            
            except Exception as e:
                print(f"❌ Error preparando datasets CNN: {e}")
                import traceback
                traceback.print_exc()

        # ========================================
        # OPCIÓN 7–8: Entrenar modelos CNN
        # ========================================
        elif choice == '7':
            print("🚧 Entrenamiento de ignición robusto — listo para ejecutar")
            # Entrenamiento aquí...

        elif choice == '8':
            print("🚧 Entrenamiento de propagación robusto — listo para ejecutar")

        elif choice == '0':
            print("👋 ¡Hasta luego!")
            break

        else:
            if choice in ['1', '2', '3', '4', '5'] and not TABULAR_AVAILABLE:
                print("❌ Módulos tabulares no disponibles")
            else:
                print("❌ Opción no válida")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Programa interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()