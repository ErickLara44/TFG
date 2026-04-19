# Resumen de Refactorización y Optimización - IberFire

Este documento detalla los cambios estructurales, correcciones y optimizaciones realizadas en el proyecto `iberfire` para mejorar su mantenibilidad, portabilidad y rendimiento.

## 1. Reestructuración del Proyecto

### 🔄 Cambios
- **Creación de paquete `src`**: Se movió todo el código fuente (scripts de datos y modelos) dentro de un directorio `src/`.
- **`__init__.py`**: Se añadieron archivos `__init__.py` para convertir los directorios en paquetes Python importables.

### 💡 ¿Por qué?
- **Modularidad**: Permite importar módulos (ej. `from src.models import ...`) de forma limpia desde cualquier script.
- **Evitar conflictos**: Aísla el código del proyecto de scripts de raíz o nombres genéricos.
- **Estándar**: Sigue las mejores prácticas de estructura de proyectos de Data Science/ML.

## 2. Gestión de Configuración y Rutas

### 🔄 Cambios
- **`src/config.py` centralizado**: Se consolidaron todas las configuraciones (rutas, hiperparámetros, variables) en un solo archivo.
- **Rutas Dinámicas (`pathlib`)**:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parents[1]
  DATA_DIR = PROJECT_ROOT / "data"
  ```
- **Variables de Entorno**: Uso de `python-dotenv` para claves API y rutas locales específicas.

### 💡 ¿Por qué?
- **Portabilidad**: El código funciona en cualquier ordenador (Mac de Erick, servidores, etc.) sin cambiar rutas "hardcoded" (ej. `/Users/erick...`).
- **Mantenibilidad**: Si cambia un parámetro del modelo, solo se edita en un lugar.

## 3. Optimización del Pipeline de Datos (Crítico)

### 🚨 El Problema
El entrenamiento original intentaba cargar mapas completos de España (920x1188 píxeles) para cada paso temporal.
- **Peso**: ~500 MB por muestra.
- **Velocidad**: ~21 segundos por muestra.
- **Inviabilidad**: Entrenar requeriría TBs de RAM y semanas de tiempo.

### 🚀 La Solución: Entrenamiento por Parches (Patch-based)
Implementamos una estrategia de **pre-computación de parches**:
1.  **Recortes Espaciales**: En lugar del mapa entero, el modelo entrena con recortes de **128x128 píxeles** centrados en zonas de interés (fuego/no fuego).
2.  **Pre-computación (`.pt`)**: Los datos se procesan *antes* de entrenar y se guardan como tensores de PyTorch listos para usar.

### ⚡ Optimizaciones de Código
- **Escaneo por Chunks**:
    - *Antes*: Leer `is_fire` día a día tardaba **~25 minutos** en escanear el dataset.
    - *Ahora*: Leer en bloques (chunks) de 100 días reduce el tiempo a **< 1 minuto**.
- **Lectura Slicing**:
    - *Antes*: Leer 7 días de contexto requería 7 lecturas al disco por variable.
    - *Ahora*: Se lee un solo "slice" temporal (`time=slice(t, t+7)`), reduciendo las operaciones de I/O drásticamente.

## 4. Nuevos Scripts

### `scripts/precompute_data.py`
Script dedicado a generar los datasets.
- Escanea el datacube buscando incendios.
- Genera parches balanceados (50% fuego, 50% no fuego).
- Guarda archivos `.pt` en `data/processed/patches/`.

### `src/train.py`
Nuevo script de entrenamiento limpio y robusto.
- Usa `PrecomputedIgnitionDataset` para carga instantánea de datos.
- Implementa `RobustFireIgnitionModel`.
- Incluye validación, guardado de mejores modelos y logging.

## Estado Actual
- **Pre-computación**: En proceso (generando ~2000 muestras de entrenamiento).
- **Entrenamiento**: Verificado y funcional. Listo para ejecutar una vez termine la generación de datos.

---
**Siguientes Pasos:**
Ejecutar el entrenamiento cuando termine la pre-computación:
```bash
.venv/bin/python src/train.py
```
