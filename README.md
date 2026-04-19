# IberFire – Sistema Integral de Predicción y Análisis de Incendios

IberFire es el resultado del Trabajo Fin de Grado enfocado en la predicción de ignición y propagación de incendios forestales en la Península Ibérica. El proyecto combina **modelos tabulares** (árboles de gradiente) con **modelos CNN robustos** basados en ConvLSTM multifrecuencia. Todo el pipeline opera sobre un **datacubo NetCDF** que integra variables topográficas, meteorológicas, índices de vegetación y datos antropogénicos.

---

## Tabla de Contenidos
- [Características Clave](#características-clave)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos y Entorno](#requisitos-y-entorno)
- [Datos de Entrada](#datos-de-entrada)
- [Modelos Implementados](#modelos-implementados)
- [Uso Rápido](#uso-rápido)
- [Flujos de Trabajo](#flujos-de-trabajo)
  - [Pipeline Tabular](#pipeline-tabular)
  - [Pipeline CNN](#pipeline-cnn)
- [Salidas y Resultados](#salidas-y-resultados)
- [Validación y Métricas](#validación-y-métricas)
- [Resolución de Problemas](#resolución-de-problemas)

---

## Características Clave
- Extracción masiva de datos desde un *datacubo* NetCDF (`data/IberFire.nc`), con filtrado territorial por CCAA de mayor incidencia.
- Pipeline tabular automatizado:
  - Generación de parquet por año.
  - Split temporal estricto Train/Val/Test.
  - Balanceo de clases configurable y normalización Z-score.
- Suite de modelos clásicos (`XGBoost`, `LightGBM`, `CatBoost`, `RandomForest`) con persistencia de pesos y feature importance.
- Datasets específicos para **Ignición** y **Propagación**, con soporte para canales estáticos/dinámicos, contexto temporal configurable y filtros de calidad.
- Modelos CNN robustos con:
  - ConvLSTM multi-escala.
  - Atención espacial y temporal.
  - Cabezas múltiples (probabilidad, confianza, riesgo).
  - Entrenamiento con mixed precision, early stopping y schedulers combinados.
- Menú CLI interactivo en `main.py` que guía todo el proceso end-to-end.

---

## Estructura del Repositorio

```
├── main.py                       # Entrada principal con menú interactivo
├── data/
│   ├── IberFire.nc               # Datacubo principal (no versionado)
│   ├── FinalDataset.csv          # Dataset plano histórico
│   ├── data_tab.py               # Pipeline tabular completo
│   ├── data_ignition_improved.py # Dataset para ignición
│   └── data_prop_improved.py     # Dataset para propagación
├── models/
│   ├── XGBoost.py                # Modelos tabulares y métricas comunes
│   ├── ignition.py               # Modelo robusto de ignición + trainer
│   └── prop.py                   # Modelo robusto de propagación + trainer
├── notebooks/                    # Análisis exploratorios y validación
├── outputs/                      # Checkpoints, métricas y mapas generados
├── reports/                      # Informes y figuras principales
├── environment.yml               # Entorno de Conda completo
└── pyproject.toml / setup.cfg    # Configuración del paquete y lint
```

> Nota: algunos GeoJSON y figuras (`reports/`, `*.png`) se usan para visualizaciones finales y validaciones espaciales.

---

## Requisitos y Entorno

- Python 3.10+ (recomendado Conda)
- PyTorch (CPU/MPS/GPU), xarray, pandas, scikit-learn
- XGBoost, LightGBM, CatBoost
- tqdm, matplotlib, geopandas (para validaciones espaciales)

Instalación rápida:

```bash
conda env create -f environment.yml
conda activate iberfire
```

O con `environment_from_history.yml` si prefieres el histórico exacto del entorno.

---

## Datos de Entrada

- `data/IberFire.nc`: datacubo multidimensional con ejes `(time, y, x)` que incluye:
  - Variables estáticas: topografía, proporciones CORINE 2018, densidad de población, distancia a carreteras, etc.
  - Variables dinámicas: temperatura, humedad, viento, precipitación, NDVI, SWI, FWI, LST.
  - Máscaras auxiliares: `is_spain`, `AutonomousCommunities`, `is_near_fire`, `is_holiday`.
- Los scripts esperan este archivo en la ruta definida en `main.py` (`datacube_path`). Ajusta la ruta si trabajas en otro entorno.

> El repo no incluye el NetCDF por su tamaño. Debes colocarlo manualmente en `data/`.

---

## Modelos Implementados

| Tipo         | Archivo                | Descripción breve |
|--------------|------------------------|-------------------|
| Tabular      | `models/XGBoost.py`    | Envuelve RF, XGB, LGBM y CatBoost. Cálculo de métricas y guardado con `joblib`. |
| Ignición CNN | `models/ignition.py`   | ConvLSTM multi-escala + atención espacial/temporal + múltiples cabezas. |
| Propagación  | `models/prop.py`       | ConvLSTM 3D con foco en mapas de fuego t+1 y pérdidas focales. |

Todos los modelos tabulares comparten la clase base `BaseFirePredictor` que gestiona carga/guardado y feature importance.

---

## Uso Rápido

1. **Configura el entorno** y asegúrate de tener el NetCDF en `data/IberFire.nc`.
2. Ejecuta el menú principal:

```bash
python main.py
```

3. Selecciona las opciones según necesites:
   - `1`: Preparar dataset tabular (crea parquet + splits + normalización).
   - `2-5`: Entrenar modelos tabulares (XGBoost, LightGBM, CatBoost, RandomForest).
   - `6`: Preparar datasets CNN (Ignición + Propagación) y DataLoaders.
   - `7`: Entrenar modelo robusto de Ignición (pendiente de conectar al trainer).
   - `8`: Entrenar modelo robusto de Propagación (pendiente de conectar).
   - `9`: Pipeline completo (Ignición → Propagación) – reservado para integración final.
   - `0`: Salir.

El menú valida dependencias: si los módulos tabulares no se pueden importar, sólo verás las opciones CNN.

---

## Flujos de Trabajo

### Pipeline Tabular
1. **Extracción (`extract_raw_data`)**  
   - Carga el NetCDF vía `xarray`.
   - Filtra por fechas (2017–2024) y por CCAA más relevantes.
   - Procesa por *chunks* temporales para controlar RAM.
   - Genera `data/processed/by_year/raw_<year>.parquet`.

2. **División y balanceo (`balance_and_split`)**  
   - Split temporal: Train ≤2022, Val 2023, Test 2024.
   - Balanceo independiente por split para evitar leakage.
   - Guarda `train/val/test.parquet`.

3. **Normalización (`normalize_data`)**  
   - Ajusta `StandardScaler` sobre Train.
   - Aplica a Val/Test y guarda el scaler (`scaler.pkl`).

4. **Entrenamiento tabular (`models/XGBoost.py`)**  
   - Cada modelo imprime métricas y guarda pesos + importancia de features.

### Pipeline CNN
1. **Creación de splits temporales** con `create_train_val_test_split`.
2. **Datasets especializados**:
   - `IgnitionDataset`: salida binaria (hay fuego en t+1). Soporta balanceo interno y diferentes modos (`cnn` vs `convlstm`).
   - `SpreadDataset`: salida raster con mapa de fuego futuro; incluye opcionalmente el canal `fire_state`.
3. **DataLoaders optimizados**: batch dual (`batch_sizes={'ignition': 32, 'spread': 16}` configurables).
4. **Entrenamiento**:
   - `train_robust_ignition_model`: usa AdamW, schedulers Cosine + ReduceLROnPlateau, mixed precision y early stopping.
   - `train_robust_spread_model`: lógica análoga para propagación (ver `models/prop.py`).

> Las opciones 7–9 del menú están preparadas para invocar los trainers; actualmente muestran un mensaje y debes llamarlos desde un notebook/script mientras ajustas hiperparámetros.

---

## Salidas y Resultados

- **Modelos guardados** en `models/Spain<Modelo>.pkl` + `_feat_importance.pkl`.
- **Métricas CNN** serializadas (`robust_ignition_metrics.json`, etc.).
- **Figuras** en `reports/` y `validacion_series_temporales_compacto.png`.
- **Visualizaciones** como `fires_10days.gif` o `mapa_area_quemada_simple.png`.
- **Tablas procesadas** en `data/processed/`.

---

## Validación y Métricas

- Métricas tabulares: sensibilidad, especificidad, AUROC, precisión, recall, F1.
- Métricas CNN:
  - AUROC y F1 optimizado mediante barrido del umbral.
  - Curvas Precision-Recall.
  - Distribución de atención espacial/temporal para interpretabilidad.
- Se recomienda registrar experimentos en `notebooks/` o herramientas externas (Weights & Biases, MLflow) si escalas entrenamientos.

---

## Resolución de Problemas

- **ImportError en opciones 1–5**: verifica dependencias de `models/XGBoost.py` (xgboost, lightgbm, catboost). Instala los paquetes faltantes o trabaja con `TABULAR_AVAILABLE=False`.
- **Falta `IberFire.nc`**: coloca el archivo en `data/` y actualiza la ruta en `main.py` si usas otra ubicación.
- **Memoria insuficiente durante `extract_raw_data`**: aumenta `chunk_size_days` gradualmente (ej. 30 → 15) o filtra más CCAA.
- **Entrenamiento CNN lento**: revisa el dispositivo detectado (CPU, MPS, CUDA) y ajusta `batch_sizes`. Puedes desactivar `use_mixed_precision` en CPU.
- **Datasets desbalanceados**: activa `balance_classes=True` en `IgnitionDataset` o ajusta `neg_pos_ratio` en el pipeline tabular.

---

## Contacto

Proyecto desarrollado por **Erick Mollinedo Lara** dentro del TFG. Para dudas o colaboración, puedes abrir *issues* en este repositorio o contactar directamente.

¡Gracias por usar IberFire! Mantén la seguridad forestal como prioridad. 🌲🔥
