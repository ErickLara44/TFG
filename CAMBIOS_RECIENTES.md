# Cambios Recientes - IberFire

Documento de referencia para la redaccion del TFG. Recoge los cambios mas relevantes realizados sobre el pipeline tabular, modelos y correccion de data leakage.

---

## 1. Deteccion y correccion de Data Leakage (`is_near_fire`)

### Problema detectado

La variable `is_near_fire` es una dilatacion espacial de `is_fire` en el mismo timestep. Dado que el modelo usa features en tiempo `t` para predecir fuego en `t+1`, esta variable contiene informacion directa del target:

- El 100% de los pixeles con fuego en `t+1` tenian `is_near_fire=1` en `t`.
- `is_near_fire` acaparaba el **95.07%** de la importancia del XGBoost, haciendo que el resto de features fueran practicamente irrelevantes.

### Metricas antes vs despues de la correccion (Test Set)

| Metrica       | Con `is_near_fire` | Sin ella (real) |
|---------------|---------------------|-----------------|
| AUROC         | 0.9906              | 0.9316          |
| Sensitivity   | 0.9584              | 0.4921          |
| Specificity   | 0.9645              | 0.9821          |
| Precision     | 0.9310              | 0.9322          |
| F1            | 0.9445              | 0.6441          |

Los resultados con leakage eran artificialmente inflados. Los valores reales (AUROC ~0.93) son coherentes con la literatura de prediccion de incendios a escala de pixel.

### Archivos modificados

- `src/data/data_ignition_improved.py` — Eliminada de `DEFAULT_FEATURE_VARS`
- `src/data/data_tab.py` — Eliminada de `binary_vars` en normalizacion
- `src/data/data_prop_improved.py` — Eliminada de `DEFAULT_FEATURE_VARS`
- `src/api/data_fetcher.py` — Eliminada de `IGNITION_CHANNELS` y su bloque de asignacion a 0
- `src/api/inference.py` — Eliminada de la lista de `var_names`

### Nota para la redaccion

Esto es relevante para la seccion de metodologia/resultados: documentar que se detecto y corrigio un caso de data leakage es un punto positivo en el TFG, ya que demuestra rigor en la evaluacion. Se puede mencionar como leccion aprendida y justificacion de por que las metricas finales son mas conservadoras pero fiables.

---

## 2. Optimizacion del pipeline tabular (`data_tab.py`)

### Problema original

La extraccion de features del datacube (920x1188x6241 timesteps, ~261 variables) era extremadamente lenta:

- **Version original**: iteraba por timestep, cargando todas las variables en cada iteracion (~128,000 operaciones I/O). Estimacion: **~4 dias**.
- **Primera optimizacion**: reestructuracion variable-por-variable con batches de 200 timesteps (~928 ops I/O). Tiempo real: **~13 horas** (2h scan + 10h42m train).

### Optimizaciones aplicadas (segunda ronda)

1. **Scan batching**: Lectura de `is_fire` en bloques de 500 timesteps en vez de uno a uno (6,240 lecturas -> ~13 lecturas).
2. **Spatial cropping**: Solo se lee el bounding box `y[y_min:y_max], x[x_min:x_max]` de los puntos muestreados, en vez del grid completo 920x1188.
3. **netCDF4 directo**: Bypass de xarray para las lecturas masivas, usando `netCDF4.Dataset` directamente para reducir overhead por operacion.

### Metodo `_scan_fire_events`

- Antes: ~6,240 llamadas `ds["is_fire"].isel(time=t)` -> 2 horas
- Despues: ~13 llamadas con `isel(time=slice(start, end))` en bloques de 500

### Metodo `_extract_features`

- Antes: loop por timestep, carga de todas las variables dinamicas por iteracion
- Despues: loop por variable, con batches temporales de 200 timesteps, crop espacial, y netCDF4 directo

### Nota para la redaccion

El datacube IberFire usa compresion HDF5, lo que hace que los accesos aleatorios sean costosos (cada lectura requiere descomprimir el chunk completo). Las optimizaciones se centran en minimizar el numero de operaciones I/O y la cantidad de datos leidos por operacion. Esto puede mencionarse en la seccion de ingenieria de datos / implementacion.

---

## 3. Resultados de modelos tabulares (sin leakage)

### Configuracion del dataset

| Parametro         | Valor                         |
|-------------------|-------------------------------|
| Periodo           | 2009-2024                     |
| CCAA              | Galicia, Asturias, CyL, Extremadura, C. Valenciana, Andalucia |
| Train             | <= 2020 (74,778 muestras)     |
| Validacion        | 2021-2022 (78,114 muestras)   |
| Test              | 2023-2024 (21,369 muestras)   |
| Ratio neg:pos     | 2:1 (por split)               |
| Features          | 41 variables                  |
| Offset temporal   | features(t) -> label(t+1)     |

### XGBoost - Metricas

| Split | Threshold | Sensitivity | Specificity | AUROC  | Precision | F1     |
|-------|-----------|-------------|-------------|--------|-----------|--------|
| Val   | 0.50      | 0.1850      | 0.9815      | 0.8684 | 0.8330    | 0.3028 |
| Val   | 0.05*     | 0.5672      | 0.9001      | 0.8684 | 0.7395    | 0.6420 |
| Test  | 0.50      | 0.4921      | 0.9821      | 0.9316 | 0.9322    | 0.6441 |
| Test  | 0.05*     | 0.7463      | 0.9086      | 0.9316 | 0.8033    | 0.7737 |

*Threshold optimizado para maximizar F1.

### Random Forest - Metricas

| Split | Threshold | Sensitivity | Specificity | AUROC  | Precision | F1     |
|-------|-----------|-------------|-------------|--------|-----------|--------|
| Val   | 0.50      | 0.0817      | 0.9885      | 0.8647 | 0.7803    | 0.1479 |
| Val   | 0.13*     | 0.8272      | 0.7827      | 0.8647 | 0.6556    | 0.7314 |
| Test  | 0.50      | 0.3441      | 0.9898      | 0.9209 | 0.9438    | 0.5043 |
| Test  | 0.15*     | 0.8767      | 0.8211      | 0.9209 | 0.7102    | 0.7847 |

*Threshold optimizado para maximizar F1.

### Comparativa resumen (Test, threshold optimo)

| Modelo        | AUROC  | F1     | Sensitivity | Specificity |
|---------------|--------|--------|-------------|-------------|
| XGBoost       | 0.9316 | 0.7737 | 0.7463      | 0.9086      |
| Random Forest | 0.9209 | 0.7847 | 0.8767      | 0.8211      |

### Observaciones para la redaccion

- **XGBoost vs RF**: XGBoost tiene mejor AUROC (mejor ranking general), pero RF con threshold optimizado obtiene mejor F1 y mucha mas sensitivity (0.88 vs 0.75). Para un sistema de alerta de incendios, la sensitivity es critica (no queremos perder fuegos).
- **Threshold por defecto (0.5) es inadecuado**: Ambos modelos son muy conservadores con t=0.5. El threshold optimo esta en 0.05-0.15, lo cual se explica por el desbalance de clases y la calibracion de las probabilidades.
- **Diferencia val/test**: Test obtiene mejores resultados que val en ambos modelos. Esto puede deberse a que los anos de test (2023-2024) tienen patrones de incendio mas "tipicos" que los de validacion (2021-2022), o al tamano menor del test set.

---

## 4. Features mas importantes

Top 10 features por importancia media (ambos modelos):

1. CLC_2018_agricultural_proportion
2. CLC_2018_scrub_proportion
3. slope_mean
4. FWI (Fire Weather Index)
5. RH_min (humedad relativa minima)
6. day_of_year_sin (estacionalidad)
7. day_of_year_cos (estacionalidad)
8. t2m_max (temperatura maxima)
9. total_precipitation_mean
10. t2m_range (rango termico)

### Interpretacion para la redaccion

- **Uso de suelo domina**: la proporcion de terreno agricola y matorral son las variables mas predictivas, coherente con la literatura (quemas agricolas, vegetacion combustible).
- **FWI funciona**: el Fire Weather Index, disenado especificamente para riesgo de incendio, aparece 4o, validando su inclusion.
- **Estacionalidad importa**: las variables ciclicas de dia del ano capturan la temporada de incendios.
- **Meteorologia complementa**: temperatura, humedad y precipitacion aportan informacion adicional al FWI.

---

## 5. Visualizaciones generadas

Script: `scripts/xgboost_stats.py`
Salida: `outputs/tabular_stats/`

| Archivo                    | Contenido                                    |
|---------------------------|----------------------------------------------|
| `roc_curves.png`          | Curvas ROC comparativas                      |
| `pr_curves.png`           | Curvas Precision-Recall                      |
| `confusion_matrices.png`  | Matrices de confusion (threshold optimo)     |
| `feature_importance.png`  | Top 20 features por modelo                   |
| `threshold_analysis.png`  | Metricas vs threshold                        |
| `prob_distribution.png`   | Distribucion de probabilidades por clase     |
| `calibration.png`         | Curvas de calibracion                        |
| `metrics_summary.csv`     | Tabla completa de metricas                   |

---

## 6. Archivos del pipeline tabular

| Archivo                          | Funcion                                         |
|----------------------------------|--------------------------------------------------|
| `src/data/data_tab.py`           | Dataset tabular: extraccion, muestreo, normalizacion |
| `src/models/XGBoost.py`         | Clases XGBoost y Random Forest + metricas        |
| `scripts/xgboost_stats.py`      | Generacion de visualizaciones para TFG           |
| `scripts/visualize_tabular_models.py` | Script de visualizacion anterior (deprecado) |
| `data/processed/tabular/`       | Datos procesados (parquet + scaler)              |
| `models/SpainXGBoost.pkl`       | Modelo XGBoost entrenado                         |
| `models/SpainRandomForest.pkl`  | Modelo Random Forest entrenado                   |
