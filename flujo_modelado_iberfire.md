# Documentación del Modelo de Ignición: Proyecto Iberfire

Este documento detalla en exclusividad el principio teórico y el ciclo completo de vida computacional (ingesta de datos, diseño arquitectónico, entrenamiento y predicción) del Modelo de Riesgo de Ignición (Iberfire).

---

## 1. Principio Físico/Teórico de la Ignición

El modelo no se basa simplemente en correlaciones estadísticas burdas, sino que fundamenta sus bases de aprendizaje sobre un **principio empírico-físico espacio-temporal**. Asume que un incendio de grandes magnitudes se desata condicionado al estrés fisiológico, meteorología extrema y actividad humana adyacente. Esta teoría se materializa directamente en la recopilación de datos (Dataset) a través de estrictos **filtros físicos limitantes** combinados vectorialmente.

Para asegurar que el modelo aprende sobre bases realistas, se impone antes el cumplimiento de reglas termodinámicas y forestales mínimas en las áreas reportadas:
1. **Bio-estrés climático:** Se excluyen del aprendizaje zonas con riesgo biológico inexistente, exigiendo un pre-calentamiento reflejado en un Fire Weather Index (**FWI ≥ 1.0**).
2. **Nivel Freático y Saturación Superficial:** El terreno no debe estar empapado, ya que el agua actuaría de sumidero térmico impidiendo la chispa base. Se garantiza el filtro exigiendo un Soil Water Index bajo (**SWI_001 ≤ 80%**).
3. **Mantenimiento Favorable:** La humedad del aire debe ser propicia para la prolongación. El oxígeno debe fluir en ausencia de vapor condensado saturante (**RH mínima ≤ 90%**).

Al proveer a la red re-correncial profunda una secuencia temporal de días ($t-T, \dots, t-1$), el principio abordado dicta que **una ignición letal es el producto de un estrés ambiental y secado gradual prolongado** en el ecosistema, permitiendo capturar así la historia climática reciente del terreno y el proceso disipativo paulatino.

---

## 2. Ingesta y Preprocesamiento de Datos
* **Script Principal:** `scripts/precompute_data.py` en uso con `data_ignition_improved.py`
* **Proceso:** 
  - Se cargan 18 macro-canales consolidados desde el `Datacube` de NetCDF (incluyendo distancias poblacionales, altitud, slope, proporción CLC de vegetación, temperatura a 2m, precipitaciones, entre otros).
  - El escaneo realiza "Chunking" mensual validando los fuegos de los satélites frente a las reglas numéricas restrictivas mencionadas.
  - Una matriz equilibrada de muestras es definida entre "Positivos Críticos (Fuego)" aleatorizados contra "Negativos Globales (Sin fuego)". 
  - Para cada índice de suceso extraído se recorta un "parche" bi-dimensional espacial (`[T, C, H, W]`). Estos cortes (por defecto a contexto $64\times 64$) son serializados y guardados en memoria solida en tensores de formato `.pt` en el directorio de `processed`. Esta metodología aumenta exorbitantemente la agilidad iterativa a la hora de inyectar las carpetas a los chips GPU/MPS durante el entrenamiento.

---

## 3. Arquitectura Robusta del Modelo
* **Clase Principal:** `RobustFireIgnitionModel` (ubicada en `src/models/ignition.py`).
* **Componentes Clave de Percepción:**
  - **`MultiScaleConvLSTM`:** Capta dependencias espaciales topográficas complejas y su evolución de sequedad cronológica a la vez, comprimiendo mediante `AvgPool` en escalas (1x, 2x, 4x) para integrar tanto el relieve general circundante como el pixel exacto donde comenzó el foco de ignición.
  - **Módulo de Atención Espacial (`SpatialAttention`):** Enseña a la red, a partir de integrales convolucionales y un "atractor Sigmoide", a multiplicar su agudeza sobre regiones candentes del parche y restarle importancia al horizonte inservible de fondo.
  - **Atención Temporal (`TemporalAttention`):** Dedica un factor flotante o "peso de consciencia" decidiendo qué día pasado importaba más en la causa encadenante de la alarma.
  - **Módulo de Triple Salida:** La salida multi-variable final se diversifica desde una cabecera oculta lineal extrayendo predicciones dispares: Nivel de Probabilidad general (`sigmoid`), Confianza estricta de su veracidad predicha, y una discretización jerárquica de Riesgo general (`Bajo`, `Medio`, `Alto`), lista para visualizaciones cartográficas.

---

## 4. Flujo de Entrenamiento
* **Script de Orquestamiento:** `scripts/train_ignition.py`
* **Mecánica Optimización:**
  - El guion inicializa clases lectoras, derivando `mean` y `std` de un muestreo rápido temporal para establecer la normalización sin perturbar distribuciones espaciales.
  - Para corregir la inherente clase desbalanceada (puesto que geográficamente hay muchísimos "ceros"), la función introduce una calibración manual del ponderador `pos_weight`.
  - **Hard Negative Mining (OHEM):** Como mitigante activo contra falsos positivos persistentes, el calculador prioriza y enfoca asimétricamente el BCE (`BCEWithLogitsLoss`) obligando al optimizador a machacar exhaustivamente el top de fallos perjudiciales y píxeles engañosos.
  - Para una refinamiento minucioso se programa el learning rate con escalones adaptativos (Cosine Annealing y decaída de mesetas), cerrando con tolerancias para salvaguardas limitantes de época o detenciones tempranas (`Early Stopping` fundamentado sobre `F1 Score`).

---

## 5. Predicción, Explicabilidad y Toma de Decisiones
* **Scripts de Evaluación:** `scripts/evaluate_ignition.py` y `analyze_ignition_shap` acoplado nativamente en el módulo.
* **Flujo Operativo (Forward):** Al introducir mediante `predict_proba` un mini-lote de matrices climáticas consecutivas a la memoria de la red adiestrada, se retornan perfiles de umbral mapeados 0-1.
* **Componente Analítico Avanzado (SHAP):** La arquitectura Iberfire va un paso más allá de crear cajas negras, empleando la teoría de juegos a través de `shap.GradientExplainer`. Mediante el cruce del vector predicho con un fondo nulo del background, infiere en tiempo real el porcentaje culpable y modificativo en términos absolutos para cada canal; empaquetando visualmente los diagramas "Beeswarm" que describen la importancia causal detallada que conllevó a un desenlace de alerta.
