# main.py
import torch
from torch.utils.data import DataLoader
import xarray as xr

# ========================================
# IMPORTACIONES CORREGIDAS
# ========================================

import sys
from pathlib import Path

# Añadir src al path si no está (para ejecución directa)
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src import config

# --- Tabular ---
try:
    from src.data.data_tab import SpainCubeFireDataset
    from src.models.XGBoost import (
        SpainXGBoostPredictor,
        SpainRandomForestPredictor,
        calculate_fire_metrics,
        BaseFirePredictor
    )
    TABULAR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Módulos tabulares no encontrados: {e}")
    TABULAR_AVAILABLE = False

# --- CNN Ignición ---
from src.data.data_ignition_improved import IgnitionDataset, create_ignition_datasets

from src.models.ignition import RobustFireIgnitionModel, train_robust_ignition_model, analyze_ignition_shap

# --- CNN Propagación ---
from src.data.data_prop_improved import SpreadDataset, create_train_val_test_split
from src.models.prop import RobustFireSpreadModel, train_robust_spread_model


# ========================================
# FUNCIONES AUXILIARES
# ========================================

def setup_cnn_datasets(datacube, temporal_context_ign=3, temporal_context_spr=3):
    """Configura datasets robustos para ignición y propagación"""
    print("🔥 Configurando datasets CNN...")
    
    # DEFINICIÓN DE AÑOS (Configuración robusta del usuario)
    train_years = list(range(2015, 2022)) # 2015-2021 (7 años)
    val_years = [2022, 2023]              # 2022-2023 (2 años)
    test_years = [2024]                   # 2024 (1 año)

    # Importar función dinámica (está en data_prop_improved)
    from src.data.data_prop_improved import create_year_split
    
    splits = create_year_split(
        datacube, 
        train_years, val_years, test_years,
        min_temporal_context=max(temporal_context_ign, temporal_context_spr)
    )
    
    print("\n📊 Creando datasets de ignición...")
    ignition_datasets = create_ignition_datasets(
        datacube, splits, temporal_context=temporal_context_ign
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
    loader_args = {
        'num_workers': config.TRAINING_CONFIG.get('num_workers', 4),
        'pin_memory': config.TRAINING_CONFIG.get('pin_memory', True),
        'persistent_workers': config.TRAINING_CONFIG.get('persistent_workers', True) if config.TRAINING_CONFIG.get('num_workers', 0) > 0 else False
    }
    
    loaders = {}
    loaders['ignition'] = {
        'train': DataLoader(datasets['ignition']['train'], batch_size=batch_sizes['ignition'], shuffle=True, **loader_args),
        'val': DataLoader(datasets['ignition']['val'], batch_size=batch_sizes['ignition'], shuffle=False, **loader_args),
        'test': DataLoader(datasets['ignition']['test'], batch_size=batch_sizes['ignition'], shuffle=False, **loader_args)
    }
    
    loaders['spread'] = {
        'train': DataLoader(datasets['spread']['train'], batch_size=batch_sizes['spread'], shuffle=True, **loader_args),
        'val': DataLoader(datasets['spread']['val'], batch_size=batch_sizes['spread'], shuffle=False, **loader_args),
        'test': DataLoader(datasets['spread']['test'], batch_size=batch_sizes['spread'], shuffle=False, **loader_args)
    }
    
    return loaders


# ========================================
# MAIN
# ========================================

def main():
    datacube_path = config.DATACUBE_PATH  # ✅ Ruta dinámica desde config
    
    # Variables de estado
    model = None          # Modelo de ignición
    model_spread = None   # Modelo de propagación

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
            print("MODELOS TABULARES:")
            print("1: Preparar dataset tabular")
            print("2: Entrenar XGBoost")
            print("3: Entrenar Random Forest\n")
        
        print("🧠 MODELOS CNN ROBUSTOS:")
        print("6: Preparar datasets CNN (Ignición + Propagación)")
        print("7: Entrenar modelo robusto de IGNICIÓN")
        print("8: Entrenar modelo robusto de PROPAGACIÓN")

        print("9: Entrenar PIPELINE COMPLETO (Ignición → Propagación)")
        print("10: Analisis SHAP (Ignición)")
        print("\n0: Salir")
        print("-"*50)
        
        choice = input("Selecciona una opción: ").strip()

        # ========================================
        # OPCIÓN 1: Preparar dataset tabular
        # ========================================
        if choice == '1' and TABULAR_AVAILABLE:
            try:
                dataset_tabular = SpainCubeFireDataset(
                    datacube_path,
                    output_dir="data/processed/tabular",
                )

                print("🚀 Construyendo dataset tabular...")
                dataset_tabular.build_dataset()

                print("📏 Normalizando...")
                dataset_tabular.normalize_data()

                print("📦 Cargando datos preparados...")
                train_X, train_y, _ = dataset_tabular.get_features_labels("train")
                val_X, val_y, _ = dataset_tabular.get_features_labels("val")
                test_X, test_y, _ = dataset_tabular.get_features_labels("test")

                if train_X is not None:
                    print(f"✅ Dataset tabular listo: Train={len(train_y):,}, Val={len(val_y):,}, Test={len(test_y):,}")

            except Exception as e:
                print(f"❌ Error preparando dataset tabular: {e}")

        # ========================================
        # OPCIÓN 2: Entrenar XGBoost
        # ========================================
        elif choice == '2' and TABULAR_AVAILABLE:
            # Auto-carga si no está en memoria
            if train_X is None:
                print("⏳ Datos no en memoria. Intentando cargar desde disco...")
                try:
                    if dataset_tabular is None:
                        dataset_tabular = SpainCubeFireDataset(
                            datacube_path,
                            output_dir="data/processed/tabular",
                        )

                    train_X, train_y, _ = dataset_tabular.get_features_labels("train")
                    val_X, val_y, _ = dataset_tabular.get_features_labels("val")
                    test_X, test_y, _ = dataset_tabular.get_features_labels("test")

                    if train_X is None:
                        print("❌ No se encontraron datos procesados. Por favor ejecuta la Opción 1 primero.")
                        continue

                except Exception as e:
                    print(f"❌ Error cargando datos: {e}")
                    continue
            try:
                model = SpainXGBoostPredictor()
                model.train_with_validation_strategy(train_X, train_y, val_X, val_y)
                test_probs = model.predict(test_X)
                test_bin = (test_probs >= 0.5).astype(int)
                print("📊 Test XGBoost:", calculate_fire_metrics(test_y, test_bin, test_probs))
            except Exception as e:
                print(f"❌ Error entrenando XGBoost: {e}")

        elif choice == '3' and TABULAR_AVAILABLE:
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
            if data_loaders is None:
                print("❌ Primero prepara los datasets (opción 6)")
                continue
                
            print("🚀 Iniciando entrenamiento de IGNICIÓN ROBUSTO...")
            
            # Obtener número real de canales
            n_channels = len(data_loaders['ignition']['train'].dataset.feature_vars)
            print(f"📊 Canales detectados (Ignition): {n_channels}")

            # Inicializar modelo
            model = RobustFireIgnitionModel(
                num_input_channels=n_channels,
                temporal_context=config.MODEL_CONFIG['temporal_context'],
                hidden_dims=[64, 128]
            )
            
            # Entrenar
            history = train_robust_ignition_model(
                model=model,
                train_loader=data_loaders['ignition']['train'],
                val_loader=data_loaders['ignition']['val'],
                epochs=config.TRAINING_CONFIG['epochs'],
                lr=config.TRAINING_CONFIG['learning_rate'],
                device=config.TRAINING_CONFIG['device'],
                save_metrics_path=str(config.OUTPUTS_DIR / "ignition_metrics.json")
            )

        elif choice == '8':
            if data_loaders is None:
                print("❌ Primero prepara los datasets (opción 6)")
                continue
                
            print("🚀 Iniciando entrenamiento de PROPAGACIÓN ROBUSTO...")
            
            # Obtener número real de canales (+1 por estado de fuego)
            n_channels = len(data_loaders['spread']['train'].dataset.feature_vars) + 1
            print(f"📊 Canales detectados (Spread): {n_channels}")

            # Inicializar modelo
            model_spread = RobustFireSpreadModel(
                input_channels=n_channels, 
                hidden_dims=[32, 64, 128]
            )
            
            # Entrenar
            history = train_robust_spread_model(
                model=model_spread,
                train_loader=data_loaders['spread']['train'],
                val_loader=data_loaders['spread']['val'],
                epochs=config.TRAINING_CONFIG['epochs'],
                lr=config.TRAINING_CONFIG['learning_rate'],
                device=config.TRAINING_CONFIG['device'],
                save_metrics_path=str(config.OUTPUTS_DIR / "spread_metrics.json")
            )


        # ========================================
        # OPCIÓN 10: Análisis SHAP Ignición
        # ========================================
        elif choice == '10':
            if data_loaders is None:
                print("❌ Primero prepara los datasets (opción 6)")
                continue

            # Intentar cargar modelo si no está en memoria
            if model is None:
                model_path = "best_robust_ignition_model.pth"
                if Path(model_path).exists():
                    print(f"🔄 Cargando modelo desde {model_path}...")
                    try:
                        # Recrear arquitectura (necesitamos saber n_channels)
                        n_channels = len(data_loaders['ignition']['train'].dataset.feature_vars)
                        model = RobustFireIgnitionModel(
                            num_input_channels=n_channels,
                            temporal_context=config.MODEL_CONFIG['temporal_context'],
                            hidden_dims=[64, 128]
                        )
                        checkpoint = torch.load(model_path, map_location=config.TRAINING_CONFIG['device'], weights_only=False)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print("✅ Modelo cargado correctamente.")
                    except Exception as e:
                        print(f"❌ Error cargando modelo: {e}")
                        model = None
                else:
                    print("❌ No hay modelo entrenado en memoria ni en disco.")
                    print("👉 Ejecuta la opción 7 primero.")
                    continue
            
            if model is not None:
                analyze_ignition_shap(
                    model=model,
                    train_loader=data_loaders['ignition']['train'],
                    test_loader=data_loaders['ignition']['test'],
                    device=config.TRAINING_CONFIG['device'],
                    output_dir=str(config.OUTPUTS_DIR)
                )
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