"""
SpainCubeFireDataset - Pipeline de preparación de datos tabulares para predicción de incendios

Este módulo convierte un datacube NetCDF (formato multidimensional) en datasets
tabulares (Parquet) listos para entrenar modelos de ML como Random Forest o XGBoost.

Características principales:
- Filtra datos por comunidades autónomas específicas
- Divide temporalmente en train/val/test
- Balancea clases (fuego/no-fuego)
- Normaliza features continuas
- Guarda metadatos del proceso

Autor: [Tu nombre]
Fecha: 2025
"""

import xarray as xr  # Para leer archivos NetCDF multidimensionales
import pandas as pd  # Para manipulación de datos tabulares
import numpy as np   # Para operaciones numéricas
import os           # Para manejo de rutas y directorios
import gc           # Garbage collector para liberar memoria
import json         # Para guardar metadatos en formato JSON
from datetime import datetime  # Para timestamps
from tqdm import tqdm          # Para barras de progreso
from sklearn.preprocessing import StandardScaler  # Para normalización Z-score
import pickle  # Para serializar objetos Python (guardar el scaler)


class SpainCubeFireDataset:
    """
    Clase principal para generar datasets tabulares desde un datacube NetCDF.
    
    Pipeline completo:
    1. extract_raw_data()    → Convierte NetCDF a Parquet por año
    2. balance_and_split()   → Divide en train/val/test y balancea
    3. normalize_data()      → Normaliza features continuas
    4. save_metadata()       → Guarda info del proceso
    
    Ejemplo de uso:
        dataset = SpainCubeFireDataset(
            datacube_path="IberFire.nc",
            output_dir="data/processed",
            start_year=2017,
            end_year=2024
        )
        dataset.extract_raw_data()
        dataset.balance_and_split()
        dataset.normalize_data()
    """

    def __init__(self, datacube_path,
                 output_dir="data/processed",
                 start_year=2017, end_year=2024,
                 ccaa_target=None,
                 chunk_size_days=30,
                 neg_pos_ratio=1.2):
        """
        Inicializa el generador de datasets.
        
        Args:
            datacube_path (str): Ruta al archivo NetCDF (.nc) con el datacube
            output_dir (str): Directorio donde se guardarán los archivos Parquet
            start_year (int): Año inicial del periodo a procesar
            end_year (int): Año final del periodo a procesar
            ccaa_target (list): Códigos de comunidades autónomas a incluir.
                               Si None, usa [12, 3, 7, 11, 1] (top 5 con más incendios)
            chunk_size_days (int): Número de días a procesar en cada iteración
                                  (mayor = más RAM pero más rápido)
            neg_pos_ratio (float): Ratio de muestras negativas/positivas deseado
                                   (1.2 = 1.2 no-fuegos por cada fuego)
        """
        
        # ============================================================
        # CONFIGURACIÓN DE PARÁMETROS
        # ============================================================
        
        # Si no se especifican CCAA, usar las top 5 con más incendios históricos
        if ccaa_target is None:
            ccaa_target = [12, 3, 7, 11,10, 1]  # Galicia, Asturias, C.León, Extremadura, Andalucía
        
        # Guardar parámetros como atributos de la instancia
        self.datacube_path = datacube_path      # Ruta al .nc
        self.output_dir = output_dir            # Carpeta de salida
        self.start_year = start_year            # Año inicio (ej: 2017)
        self.end_year = end_year                # Año fin (ej: 2024)
        self.ccaa_target = ccaa_target          # Lista de códigos CCAA
        self.chunk_size_days = chunk_size_days  # Días por chunk (eficiencia vs RAM)
        self.neg_pos_ratio = neg_pos_ratio      # Ratio de balanceo

        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)

        # ============================================================
        # DEFINIR RUTAS DE ARCHIVOS DE SALIDA
        # ============================================================
        
        # Carpeta para archivos intermedios (Parquet por año)
        self.parquet_dir = os.path.join(output_dir, "by_year")
        os.makedirs(self.parquet_dir, exist_ok=True)
        
        # Archivos finales de train/val/test
        self.train_file = os.path.join(output_dir, "train.parquet")
        self.val_file   = os.path.join(output_dir, "val.parquet")
        self.test_file  = os.path.join(output_dir, "test.parquet")
        
        # Archivos de metadatos
        self.scaler_path = os.path.join(output_dir, "scaler.pkl")      # Normalizador
        self.metadata_path = os.path.join(output_dir, "dataset_metadata.json")  # Info del proceso

        # Obtener lista de features a usar
        self.feature_vars = self._get_feature_variables()
        
        # ============================================================
        # DICCIONARIO DE NOMBRES DE COMUNIDADES AUTÓNOMAS
        # ============================================================
        self.CCAA_NAMES = {
            1: "Andalucía",              # Sur de España
            3: "Asturias",               # Norte (Cantábrico)
            7: "Castilla y León",        # Noroeste (la más grande)
            10: "Comunidad Valenciana",  # Este (Mediterráneo)
            11: "Extremadura",           # Oeste (frontera Portugal)
            12: "Galicia",               # Noroeste (mayor incidencia)
            
        }
        
        # ============================================================
        # IMPRIMIR CONFIGURACIÓN INICIAL
        # ============================================================
        print("\n" + "="*60)
        print("🔥 DATASET TABULAR PARA PREDICCIÓN DE INCENDIOS")
        print("="*60)
        print(f"📁 Datacube: {datacube_path}")
        print(f"📅 Periodo: {start_year}-{end_year}")
        print(f"🌍 Comunidades autónomas seleccionadas:")
        
        # Mostrar nombre legible de cada CCAA seleccionada
        for code in self.ccaa_target:
            name = self.CCAA_NAMES.get(code, f"Desconocida ({code})")
            print(f"   • {name} (código {code})")
        
        print(f"⚖️  Ratio neg/pos objetivo: {neg_pos_ratio}")
        print("="*60 + "\n")

    def _get_feature_variables(self):
        """
        Define las variables que se usarán como features (X) del modelo.
        Estas son las columnas del datacube que contienen información predictiva
        sobre el riesgo de incendio. NO incluye la variable objetivo (is_fire).
        
        Returns:
            list: Lista de nombres de variables a extraer del datacube
        """
        return [
            # === TOPOGRAFÍA (estáticas) ===
            'elevation_mean',  # Altitud media de la celda (metros)
            'slope_mean',      # Pendiente media del terreno (grados)
            
            # === COBERTURA DEL SUELO - CORINE Land Cover 2018 (estáticas) ===
            'CLC_2018_forest_proportion',        # % de bosque en la celda
            'CLC_2018_scrub_proportion',         # % de matorral/arbustos
            'CLC_2018_agricultural_proportion',  # % de cultivos/agricultura
            'CLC_2018_artificial_proportion',
    
            
            # === VARIABLES ANTROPOGÉNICAS (estáticas/semi-estáticas) ===
            'dist_to_roads_mean',  # Distancia media a carreteras (accesibilidad)
            'popdens_2018',        # Densidad de población (habitantes/km²)
            'is_waterbody',        # ¿Es cuerpo de agua? (binaria: 0/1)
            'is_holiday',          # ¿Es día festivo? (binaria: 0/1)
            
            # === METEOROLOGÍA (dinámicas - cambian diariamente) ===
            't2m_mean',                  # Temperatura media a 2m (°C)
            'RH_min',                    # Humedad relativa mínima (%)
            'wind_speed_mean',           # Velocidad del viento media (m/s)
            'wind_direction_mean',       # Dirección del viento (grados)
            'total_precipitation_mean',  # Precipitación total (mm)
            
            # === ÍNDICES DE VEGETACIÓN Y SEQUÍA (dinámicos) ===
            'NDVI',     # Normalized Difference Vegetation Index (verdor: -1 a 1)
            'SWI_010',  # Soil Water Index a 10cm profundidad (humedad suelo: 0-100)
            'FWI',      # Fire Weather Index (peligro de incendio: 0-100+)
            'LST',      # Land Surface Temperature (temperatura superficie: °C)
            
            # === FEATURES ESPACIALES (derivadas) ===
            'is_near_fire'  # ¿Hay fuego en celdas vecinas? (binaria: 0/1)
        ]

    def extract_raw_data(self):
        """
        PASO 1: Extrae datos del datacube NetCDF y los convierte a Parquet.
        
        Proceso:
        1. Carga el datacube NetCDF (multi-GB, formato comprimido)
        2. Filtra por periodo temporal (start_year a end_year)
        3. Filtra espacialmente (solo España, solo CCAA seleccionadas)
        4. Procesa por chunks temporales para no saturar RAM
        5. Acumula DataFrames en memoria por año
        6. Guarda un archivo Parquet por año
        
        Returns:
            dict: Estadísticas de filas procesadas por año
        """
        print("🧩 PASO 1: Extrayendo datos del datacube...")
        print(f"⏱️  Esto puede tardar varios minutos...\n")
        
        # ============================================================
        # CARGAR DATACUBE
        # ============================================================
        # xr.open_dataset() carga un archivo NetCDF como un objeto Dataset
        # Similar a un diccionario de arrays multidimensionales (x, y, time)
        ds = xr.open_dataset(self.datacube_path)
        
        # ============================================================
        # FILTRAR PERIODO TEMPORAL
        # ============================================================
        # Construir fechas en formato ISO (YYYY-MM-DD)
        start_date = f"{self.start_year}-01-01"
        end_date   = f"{self.end_year}-12-31"
        
        # .sel() selecciona un subconjunto del datacube por dimensión
        # slice(start, end) define un rango continuo
        ds = ds.sel(time=slice(start_date, end_date))
        
        # ============================================================
        # FILTRAR ESPACIALMENTE: SOLO ESPAÑA
        # ============================================================
        # Si existe la variable "is_spain" (máscara binaria)
        if "is_spain" in ds:
            # .where() mantiene solo valores donde condición=True
            # drop=True elimina coordenadas completamente fuera de España
            ds = ds.where(ds["is_spain"] == 1, drop=True)
            print("✅ Filtrado espacial: solo celdas con is_spain=1")
        
        # ============================================================
        # SELECCIONAR SOLO VARIABLES NECESARIAS
        # ============================================================
        # Lista completa: features + variable objetivo + metadatos
        vars_needed = self.feature_vars + ['is_fire', 'AutonomousCommunities']
        
        # Verificar cuáles variables existen realmente en el datacube
        # (puede que falten algunas si el datacube está incompleto)
        vars_available = [v for v in vars_needed 
                         if v in ds.data_vars or v in ds.coords]
        
        # Reportar variables disponibles vs faltantes
        print(f"📊 Variables disponibles: {len(vars_available)}/{len(vars_needed)}")
        missing = set(vars_needed) - set(vars_available)
        if missing:
            print(f"⚠️  Variables faltantes: {missing}")
        
        # Reducir el datacube solo a las variables que usaremos
        # (ahorra RAM y tiempo de procesamiento)
        ds = ds[vars_available]
        
        # ============================================================
        # PREPARAR LOOP DE PROCESAMIENTO POR CHUNKS
        # ============================================================
        # Extraer array de timestamps (numpy.datetime64)
        time_index = ds.time.values
        total_days = len(time_index)  # Ej: 2922 días si procesas 2017-2024
        
        print(f"📅 Total días a procesar: {total_days}")
        print(f"🔢 Chunk size: {self.chunk_size_days} días\n")
        
        # Diccionario para acumular DataFrames por año
        # Estructura: {2017: [df1, df2, ...], 2018: [df1, df2, ...]}
        year_buffers = {}
        
        # Contadores globales
        processed_total = 0      # Total de filas procesadas
        processed_per_year = {}  # Filas por año (para estadísticas)
        
        # Crear barra de progreso con tqdm
        pbar = tqdm(total=total_days, desc="Extrayendo", unit="días")
        
        # ============================================================
        # LOOP PRINCIPAL: PROCESAR POR CHUNKS TEMPORALES
        # ============================================================
        # range(start, stop, step) genera índices para cada chunk
        # Ej: 0, 30, 60, ..., 2910 si chunk_size_days=30
        for start in range(0, total_days, self.chunk_size_days):
            # Calcular índice final del chunk (sin pasarse del total)
            end = min(start + self.chunk_size_days, total_days)
            
            # --- Cargar chunk temporal del datacube ---
            # .isel() selecciona por índices (no por valores)
            # slice(start, end) toma días desde start hasta end-1
            ds_chunk = ds.isel(time=slice(start, end))
            
            # --- Convertir xarray → pandas DataFrame ---
            # .to_dataframe() convierte el chunk a formato tabular
            # reset_index() convierte coordenadas (x,y,time) en columnas normales
            df_chunk = ds_chunk.to_dataframe().reset_index()
            
            # --- Eliminar filas con valores faltantes (NaN) ---
            # dropna(subset=...) elimina filas donde alguna variable necesaria es NaN
            # CRÍTICO: evita problemas en el entrenamiento del modelo
            df_chunk = df_chunk.dropna(subset=vars_available)
            
            # --- Filtrar por comunidades autónomas ---
            # Solo mantener filas donde AutonomousCommunities esté en la lista objetivo
            if "AutonomousCommunities" in df_chunk.columns:
                df_chunk = df_chunk[df_chunk["AutonomousCommunities"].isin(self.ccaa_target)]
            
            # Si después de filtrar no queda nada, pasar al siguiente chunk
            if len(df_chunk) == 0:
                pbar.update(end - start)  # Actualizar barra de progreso
                continue
            
            # --- Añadir columna de año ---
            # pd.to_datetime() convierte strings a datetime
            # .dt.year extrae solo el año (2024, 2023, etc.)
            df_chunk["year"] = pd.to_datetime(df_chunk["time"]).dt.year
            
            # --- Acumular DataFrames por año ---
            # Agrupar por año y añadir cada subgrupo al buffer correspondiente
            for year, df_y in df_chunk.groupby("year"):
                # Si es la primera vez que vemos este año, inicializar lista
                if year not in year_buffers:
                    year_buffers[year] = []
                    processed_per_year[year] = 0
                
                # Añadir DataFrame del año al buffer
                year_buffers[year].append(df_y)
                # Contar filas procesadas
                processed_per_year[year] += len(df_y)
            
            # --- Actualizar contadores y barra de progreso ---
            processed_total += len(df_chunk)
            pbar.set_postfix({"Filas": f"{processed_total:,}"})  # Mostrar nº filas
            pbar.update(end - start)  # Avanzar barra por días procesados
            
            # --- Liberar memoria ---
            # Eliminar objetos grandes que ya no necesitamos
            del ds_chunk, df_chunk
            # Forzar garbage collector para liberar RAM inmediatamente
            gc.collect()
        
        # Cerrar barra de progreso
        pbar.close()
        
        # ============================================================
        # GUARDAR BUFFERS ACUMULADOS EN ARCHIVOS PARQUET
        # ============================================================
        print("\n💾 Guardando archivos Parquet por año...")
        
        # Iterar sobre años en orden cronológico
        for year in sorted(year_buffers.keys()):
            # Obtener lista de DataFrames de ese año
            dfs = year_buffers[year]
            
            # Concatenar todos los DataFrames del año en uno solo
            # ignore_index=True renumera filas desde 0
            combined = pd.concat(dfs, ignore_index=True)
            
            # Construir ruta del archivo Parquet
            # Ej: data/processed/by_year/raw_2024.parquet
            parquet_path = os.path.join(self.parquet_dir, f"raw_{year}.parquet")
            
            # Guardar en formato Parquet con compresión Snappy
            # Parquet es ~10x más rápido que CSV y ocupa menos espacio
            combined.to_parquet(parquet_path, compression="snappy")
            
            # Contar cuántos fuegos hubo ese año (para estadísticas)
            n_fire = (combined['is_fire'] == 1).sum()
            print(f"   • {year}: {len(combined):,} filas ({n_fire:,} fuegos)")
        
        # Cerrar datacube para liberar recursos
        ds.close()
        
        # ============================================================
        # RESUMEN FINAL
        # ============================================================
        print(f"\n✅ Extracción completada:")
        print(f"   • Total filas: {processed_total:,}")
        print(f"   • Archivos: {self.parquet_dir}\n")
        
        # Retornar diccionario con estadísticas
        return processed_per_year

    def balance_and_split(self):
        """
        PASO 2: Divide temporalmente y balancea clases.
        
        Proceso CORRECTO (evita data leakage):
        1. Cargar todos los años procesados
        2. PRIMERO: Dividir por año (train ≤2022, val=2023, test=2024)
        3. DESPUÉS: Balancear cada split independientemente
        4. Guardar train.parquet, val.parquet, test.parquet
        
        ¿Por qué este orden?
        - Si balanceas primero, mezclas años → data leakage temporal
        - Al dividir primero, cada split mantiene su distribución temporal real
        
        Returns:
            dict: Número de muestras por split {'train': N, 'val': M, 'test': K}
        """
        print("🔪 PASO 2: División temporal y balanceo...")
        
        # ============================================================
        # CARGAR TODOS LOS ARCHIVOS PARQUET
        # ============================================================
        # Listar todos los archivos .parquet en la carpeta by_year
        files = [os.path.join(self.parquet_dir, f) 
                for f in os.listdir(self.parquet_dir) 
                if f.endswith(".parquet")]
        
        print(f"📂 Cargando {len(files)} archivos...")
        
        # Leer cada archivo Parquet y guardar en una lista
        dfs = [pd.read_parquet(f) for f in files]
        
        # Concatenar todos los años en un solo DataFrame
        df = pd.concat(dfs, ignore_index=True)
        
        # Asegurar que la columna 'year' existe y es numérica
        df["year"] = pd.to_datetime(df["time"]).dt.year
        
        print(f"📊 Total filas cargadas: {len(df):,}\n")
        
        # ============================================================
        # MOSTRAR ESTADÍSTICAS PRE-BALANCEO
        # ============================================================
        print("📈 Distribución por año (ANTES del balanceo):")
        
        # Iterar sobre cada año en orden cronológico
        for year in sorted(df['year'].unique()):
            # Filtrar datos de ese año
            df_year = df[df['year'] == year]
            
            # Contar fuegos y no-fuegos
            n_fire = (df_year['is_fire'] == 1).sum()
            n_nofire = (df_year['is_fire'] == 0).sum()
            
            # Calcular ratio desbalanceo (0 si no hay fuegos)
            ratio = n_nofire / n_fire if n_fire > 0 else 0
            
            # Imprimir estadísticas del año
            print(f"   {year}: {len(df_year):,} filas | "
                  f"Fuegos: {n_fire:,} | No-fuegos: {n_nofire:,} | "
                  f"Ratio: {ratio:.1f}")
        
        # ============================================================
        # DIVISIÓN TEMPORAL (CRÍTICO: HACER PRIMERO)
        # ============================================================
        print(f"\n🔪 División temporal:")
        
        # Train: todos los años hasta 2022 (inclusive)
        train = df[df["year"] <= 2022].copy()
        
        # Validation: solo año 2023
        val = df[df["year"] == 2023].copy()
        
        # Test: solo año 2024
        test = df[df["year"] == 2024].copy()
        
        # Reportar tamaños
        print(f"   • Train (≤2022): {len(train):,} filas")
        print(f"   • Val (2023):    {len(val):,} filas")
        print(f"   • Test (2024):   {len(test):,} filas")
        
        # ============================================================
        # FUNCIÓN DE BALANCEO (APLICAR A CADA SPLIT)
        # ============================================================
        def balance_split(df_split, ratio, split_name):
            """
            Balancea un split manteniendo el ratio neg/pos especificado.
            
            Args:
                df_split (DataFrame): Split a balancear
                ratio (float): Ratio deseado no-fuego/fuego (ej: 1.2)
                split_name (str): Nombre del split (para logging)
            
            Returns:
                DataFrame: Split balanceado y mezclado
            """
            # Separar filas con fuego y sin fuego
            fire = df_split[df_split["is_fire"] == 1]
            nofire = df_split[df_split["is_fire"] == 0]
            
            # Caso especial: si no hay fuegos, retornar vacío
            if len(fire) == 0:
                print(f"⚠️  {split_name}: No hay fuegos, devolviendo vacío")
                return pd.DataFrame()
            
            # Calcular cuántas muestras negativas necesitamos
            # Ej: 100 fuegos * 1.2 = 120 no-fuegos
            n_nofire_needed = int(len(fire) * ratio)
            
            # No podemos tomar más no-fuegos de los que hay disponibles
            n_nofire_actual = min(n_nofire_needed, len(nofire))
            
            # Advertir si no hay suficientes no-fuegos
            if n_nofire_actual < len(fire):
                print(f"⚠️  {split_name}: No hay suficientes no-fuegos "
                      f"(queremos {n_nofire_needed:,}, tenemos {len(nofire):,})")
            
            # Muestrear aleatoriamente no-fuegos (con seed fija para reproducibilidad)
            nofire_sampled = nofire.sample(n=n_nofire_actual, random_state=42)
            
            # Concatenar fuegos + no-fuegos muestreados
            balanced = pd.concat([fire, nofire_sampled], ignore_index=True)
            
            # Mezclar filas aleatoriamente (evita que modelo aprenda orden)
            balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Calcular ratio real conseguido
            actual_ratio = n_nofire_actual / len(fire)
            
            # Imprimir resumen del balanceo
            print(f"\n   {split_name}:")
            print(f"      Fuegos: {len(fire):,}")
            print(f"      No-fuegos: {n_nofire_actual:,}")
            print(f"      Ratio final: {actual_ratio:.2f}")
            print(f"      Total: {len(balanced):,}")
            
            return balanced
        
        # ============================================================
        # APLICAR BALANCEO A CADA SPLIT
        # ============================================================
        print(f"\n⚖️  Balanceando cada split (ratio objetivo: {self.neg_pos_ratio})...")
        
        # Balancear train, val y test independientemente
        train_balanced = balance_split(train, self.neg_pos_ratio, "TRAIN")
        val_balanced   = balance_split(val, self.neg_pos_ratio, "VAL")
        test_balanced  = balance_split(test, self.neg_pos_ratio, "TEST")
        
        # ============================================================
        # GUARDAR ARCHIVOS FINALES
        # ============================================================
        # Guardar cada split en su archivo Parquet correspondiente
        train_balanced.to_parquet(self.train_file, compression="snappy")
        val_balanced.to_parquet(self.val_file, compression="snappy")
        test_balanced.to_parquet(self.test_file, compression="snappy")
        
        print(f"\n✅ Archivos guardados:")
        print(f"   • {self.train_file}")
        print(f"   • {self.val_file}")
        print(f"   • {self.test_file}\n")
        
        # Retornar diccionario con tamaños
        return {
            'train': len(train_balanced),
            'val': len(val_balanced),
            'test': len(test_balanced)
        }

    def normalize_data(self):
        """
        PASO 3: Normaliza features continuas usando StandardScaler.
        
        ¿Qué hace StandardScaler?
        - Transforma cada feature a media=0, desviación=1
        - Fórmula: z = (x - mean) / std
        - Ejemplo: [10, 20, 30] → [-1.22, 0, 1.22]
        
        IMPORTANTE:
        - Solo normaliza variables CONTINUAS (temperatura, FWI, etc.)
        - NO normaliza binarias (is_waterbody, is_holiday, is_near_fire)
        - El scaler se ajusta SOLO en train (evita data leakage)
        - Luego se aplica el MISMO scaler a val y test
        
        ¿Por qué normalizar?
        - Random Forest no lo necesita estrictamente
        - Pero ayuda a comparar importancia de features
        - Algunos modelos (regresión logística, SVM) SÍ lo requieren
        """
        print("📏 PASO 3: Normalizando features...")
        
        # Cargar dataset de entrenamiento (sin normalizar)
        df_train = pd.read_parquet(self.train_file)
        
        # ============================================================
        # IDENTIFICAR QUÉ VARIABLES NORMALIZAR
        # ============================================================
        # Variables binarias (0/1) NO deben normalizarse
        # Razón: perderían su significado (0.5 no tiene sentido en binaria)
        binary_vars = ['is_waterbody', 'is_holiday', 'is_near_fire']
        
        # Filtrar solo features continuas que existan en el DataFrame
        cols_to_normalize = [c for c in self.feature_vars 
                            if c in df_train.columns and c not in binary_vars]
        
        # Mostrar qué se va a normalizar
        print(f"   Variables continuas a normalizar ({len(cols_to_normalize)}):")
        for col in cols_to_normalize:
            print(f"      • {col}")
        
        print(f"\n   Variables binarias (sin normalizar): {binary_vars}")
        
        # ============================================================
        # AJUSTAR SCALER EN TRAIN
        # ============================================================
        # Crear instancia del normalizador
        scaler = StandardScaler()
        
        # .fit() calcula media y std de cada columna en train
        # CRÍTICO: Solo usar train para evitar data leakage
        scaler.fit(df_train[cols_to_normalize])
        
        # ============================================================
        # GUARDAR SCALER PARA USO FUTURO
        # ============================================================
        # pickle.dump() serializa el objeto scaler a disco
        # Necesario para:
        # 1. Aplicar la misma normalización a datos nuevos en producción
        # 2. Asegurar que val/test usan las MISMAS estadísticas que train
        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"\n   💾 Scaler guardado: {self.scaler_path}")
        
        # ============================================================
        # APLICAR NORMALIZACIÓN A CADA SPLIT
        # ============================================================
        # Lista de splits a procesar: (nombre, ruta_archivo)
        for name, path in [("train", self.train_file), 
                          ("val", self.val_file), 
                          ("test", self.test_file)]:
            
            # Cargar split desde Parquet
            df = pd.read_parquet(path)
            
            # .transform() aplica la normalización usando media/std de train
            # IMPORTANTE: NO recalcula estadísticas, usa las ya guardadas
            # Esto previene data leakage (val/test no "ven" sus propias stats)
            df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
            
            # Construir ruta del archivo normalizado
            # Ej: train.parquet → train_normalized.parquet
            normalized_path = path.replace(".parquet", "_normalized.parquet")
            
            # Guardar versión normalizada
            df.to_parquet(normalized_path, compression="snappy")
            print(f"   ✅ {name.upper()} → {normalized_path}")
        
        print("")  # Línea en blanco para formato

    def save_metadata(self, extraction_stats, split_stats):
        """
        Guarda metadatos del proceso de generación del dataset.
        
        ¿Por qué guardar metadatos?
        - Reproducibilidad: saber exactamente cómo se generó el dataset
        - Debugging: si algo falla, revisar qué parámetros se usaron
        - Documentación: para otros investigadores o tu yo del futuro
        - Versionado: comparar diferentes versiones del dataset
        
        Args:
            extraction_stats (dict): Estadísticas de la extracción por año
            split_stats (dict): Tamaños de cada split después del balanceo
        """
        
        # ============================================================
        # CONSTRUIR DICCIONARIO DE METADATOS
        # ============================================================
        metadata = {
            # Timestamp de creación (formato ISO 8601)
            # Ej: "2025-10-10T14:30:45.123456"
            'creation_date': datetime.now().isoformat(),
            
            # Ruta al datacube original usado
            'datacube_path': self.datacube_path,
            
            # Periodo temporal procesado (string legible)
            'period': f"{self.start_year}-{self.end_year}",
            
            # Códigos numéricos de las CCAA incluidas
            'ccaa_codes': self.ccaa_target,
            
            # Nombres legibles de las CCAA (para humanos)
            # Ej: [12, 3] → ["Galicia", "Asturias"]
            'ccaa_names': [self.CCAA_NAMES.get(c, f"Unknown_{c}") 
                          for c in self.ccaa_target],
            
            # Ratio de balanceo aplicado
            'neg_pos_ratio': self.neg_pos_ratio,
            
            # Lista completa de features usadas
            'features': self.feature_vars,
            
            # Estadísticas del paso de extracción
            # Ej: {'2017': 142345, '2018': 156789, ...}
            'extraction_stats': extraction_stats,
            
            # Tamaños finales de cada split
            # Ej: {'train': 76047, 'val': 8234, 'test': 9876}
            'split_stats': split_stats,
            
            # Rutas a todos los archivos generados
            'files': {
                'train': self.train_file,
                'val': self.val_file,
                'test': self.test_file,
                'scaler': self.scaler_path
            }
        }
        
        # ============================================================
        # GUARDAR JSON
        # ============================================================
        # Escribir diccionario como archivo JSON formateado
        # indent=2 hace que sea legible para humanos
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📄 Metadatos guardados: {self.metadata_path}")

    def get_feature_labels(self, split="train", normalized=True):
        """
        Carga un split del dataset y lo divide en X (features), y (labels), metadata.
        
        Esta es la función que usarás para entrenar tus modelos.
        
        Args:
            split (str): Qué split cargar: "train", "val" o "test"
            normalized (bool): Si True, carga versión normalizada
                              Si False, carga versión sin normalizar
        
        Returns:
            tuple: (X, y, metadata)
                X (DataFrame): Features (shape: [N_samples, N_features])
                y (Series): Labels (0=no-fuego, 1=fuego)
                metadata (DataFrame): Columnas x, y, time (para análisis espacial)
        
        Ejemplo de uso:
            >>> X_train, y_train, meta_train = dataset.get_feature_labels("train")
            >>> X_train.shape
            (76047, 19)  # 76k muestras, 19 features
            >>> y_train.value_counts()
            0    45628  # No-fuego
            1    30419  # Fuego
        """
        
        # ============================================================
        # DETERMINAR RUTA DEL ARCHIVO
        # ============================================================
        # Sufijo según si queremos versión normalizada o no
        suffix = "_normalized.parquet" if normalized else ".parquet"
        
        # Mapa de nombres a rutas de archivos
        path_map = {
            "train": self.train_file,
            "val": self.val_file,
            "test": self.test_file
        }
        
        # Validar que el split sea válido
        if split not in path_map:
            raise ValueError(f"split debe ser 'train', 'val' o 'test', recibido: {split}")
        
        # Construir ruta completa
        # Ej: "data/processed/train.parquet" → "data/processed/train_normalized.parquet"
        path = path_map[split].replace(".parquet", suffix)
        
        # Verificar que el archivo existe
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró: {path}\n"
                                   f"¿Has ejecutado extract_raw_data() y normalize_data()?")
        
        # ============================================================
        # CARGAR DATOS
        # ============================================================
        # Leer archivo Parquet completo en memoria
        df = pd.read_parquet(path)
        
        # ============================================================
        # VERIFICAR INTEGRIDAD DE FEATURES
        # ============================================================
        # Buscar features que deberían estar pero no están
        missing = [f for f in self.feature_vars if f not in df.columns]
        if missing:
            print(f"⚠️  Features faltantes en {split}: {missing}")
            print(f"    Esto puede indicar un problema en extract_raw_data()")
        
        # Filtrar solo features que realmente existen
        available = [f for f in self.feature_vars if f in df.columns]
        
        # ============================================================
        # DIVIDIR EN X, y, METADATA
        # ============================================================
        # X: DataFrame con solo las columnas de features
        # .copy() crea una copia independiente (evita warnings de pandas)
        X = df[available].copy()
        
        # y: Series con la variable objetivo (0 o 1)
        y = df["is_fire"].copy()
        
        # metadata: Información espaciotemporal para análisis posterior
        # x, y: coordenadas espaciales (metros en proyección LAEA)
        # time: timestamp (datetime64)
        metadata = df[["x", "y", "time"]].copy()
        
        # Retornar tupla (X, y, metadata)
        return X, y, metadata
    
    def print_summary(self):
        """
        Imprime un resumen completo y legible del dataset generado.
        
        Útil para:
        - Verificar que todo se procesó correctamente
        - Ver distribución de clases en cada split
        - Confirmar rangos temporales
        - Reportar en documentación/papers
        """
        print("\n" + "="*60)
        print("📊 RESUMEN DEL DATASET")
        print("="*60 + "\n")
        
        # ============================================================
        # ITERAR SOBRE CADA SPLIT
        # ============================================================
        for split_name in ["train", "val", "test"]:
            try:
                # Cargar split (sin normalizar, para ver stats reales)
                X, y, meta = self.get_feature_labels(split_name, normalized=False)
                
                # Contar fuegos y no-fuegos
                n_fire = (y == 1).sum()
                n_nofire = (y == 0).sum()
                
                # Calcular ratio (prevenir división por cero)
                ratio = n_nofire / n_fire if n_fire > 0 else 0
                
                # Imprimir estadísticas del split
                print(f"{split_name.upper()}:")
                print(f"  • Samples: {len(y):,}")
                print(f"  • Fuegos: {n_fire:,} ({n_fire/len(y)*100:.1f}%)")
                print(f"  • No-fuegos: {n_nofire:,} ({n_nofire/len(y)*100:.1f}%)")
                print(f"  • Ratio: {ratio:.2f}")
                
                # Extraer rango temporal (min y max de la columna time)
                time_min = meta['time'].min()
                time_max = meta['time'].max()
                print(f"  • Periodo: {time_min} → {time_max}")
                
                # Número de features (columnas de X)
                print(f"  • Features: {X.shape[1]}\n")
                
            except Exception as e:
                # Si hay algún error (archivo no existe, etc.), reportarlo
                print(f"{split_name.upper()}: ⚠️  Error al cargar: {e}\n")
        
        print("="*60 + "\n")


# ============================================================================
# FUNCIÓN DE PIPELINE COMPLETO (WRAPPER DE CONVENIENCIA)
# ============================================================================
def run_full_pipeline(datacube_path, output_dir="data/processed", 
                     start_year=2017, end_year=2024,
                     ccaa_target=None, neg_pos_ratio=1.2):
    """
    Ejecuta el pipeline completo de preparación de datos en un solo comando.
    
    Esta función es un wrapper que ejecuta todos los pasos en secuencia:
    1. Instancia la clase SpainCubeFireDataset
    2. Extrae datos raw del datacube → Parquet por año
    3. Balancea y divide en train/val/test
    4. Normaliza features continuas
    5. Guarda metadatos del proceso
    6. Imprime resumen final
    
    Args:
        datacube_path (str): Ruta al archivo NetCDF
        output_dir (str): Directorio de salida
        start_year (int): Año inicial (ej: 2017)
        end_year (int): Año final (ej: 2024)
        ccaa_target (list): Lista de códigos CCAA o None para usar default
        neg_pos_ratio (float): Ratio de balanceo deseado
    
    Returns:
        SpainCubeFireDataset: Instancia del dataset (para cargar datos)
    
    Ejemplo de uso:
        >>> dataset = run_full_pipeline(
        ...     datacube_path="IberFire.nc",
        ...     output_dir="data/processed",
        ...     start_year=2017,
        ...     end_year=2024,
        ...     ccaa_target=[12, 3, 7, 11, 1],
        ...     neg_pos_ratio=1.2
        ... )
        >>> 
        >>> # Después puedes cargar los datos:
        >>> X_train, y_train, _ = dataset.get_feature_labels("train")
    """
    
    # ============================================================
    # PASO 0: CREAR INSTANCIA DEL DATASET
    # ============================================================
    dataset = SpainCubeFireDataset(
        datacube_path=datacube_path,
        output_dir=output_dir,
        start_year=start_year,
        end_year=end_year,
        ccaa_target=ccaa_target,
        neg_pos_ratio=neg_pos_ratio
    )
    
    # ============================================================
    # PASO 1: EXTRAER DATOS DEL DATACUBE
    # ============================================================
    # Convierte NetCDF → Parquet por año
    # Retorna diccionario con estadísticas de filas por año
    extraction_stats = dataset.extract_raw_data()
    
    # ============================================================
    # PASO 2: DIVIDIR Y BALANCEAR
    # ============================================================
    # Divide temporalmente (train/val/test)
    # Balancea cada split independientemente
    # Retorna diccionario con tamaños finales
    split_stats = dataset.balance_and_split()
    
    # ============================================================
    # PASO 3: NORMALIZAR FEATURES
    # ============================================================
    # Aplica StandardScaler a variables continuas
    # Guarda scaler para uso futuro
    dataset.normalize_data()
    
    # ============================================================
    # PASO 4: GUARDAR METADATOS
    # ============================================================
    # Crea archivo JSON con toda la información del proceso
    dataset.save_metadata(extraction_stats, split_stats)
    
    # ============================================================
    # PASO 5: MOSTRAR RESUMEN
    # ============================================================
    # Imprime estadísticas finales del dataset
    dataset.print_summary()
    
    # Retornar instancia del dataset para uso posterior
    return dataset


# ============================================================================
# EJEMPLO DE USO COMPLETO
# ============================================================================
if __name__ == "__main__":
    """
    Este bloque solo se ejecuta si ejecutas directamente este archivo.
    No se ejecuta si importas la clase desde otro script.
    """
    
    # Configuración del pipeline
    DATACUBE_PATH = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"  # ← Cambiar a tu ruta
    OUTPUT_DIR = "data/processed"
    START_YEAR = 2017
    END_YEAR = 2024
    
    # Top 5 CCAA con más incendios históricos
    CCAA_TARGET = [12, 3, 7, 11, 1]  # Galicia, Asturias, C.León, Extremadura, Andalucía
    
    # Ratio de balanceo (1.2 = 1.2 no-fuegos por cada fuego)
    NEG_POS_RATIO = 1.2
    
    print("🚀 Iniciando pipeline de preparación de datos...")
    print(f"📁 Datacube: {DATACUBE_PATH}")
    print(f"📅 Periodo: {START_YEAR}-{END_YEAR}")
    print(f"🌍 CCAA: {CCAA_TARGET}")
    print(f"⚖️  Ratio: {NEG_POS_RATIO}\n")
    
    # Ejecutar pipeline completo
    dataset = run_full_pipeline(
        datacube_path=DATACUBE_PATH,
        output_dir=OUTPUT_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        ccaa_target=CCAA_TARGET,
        neg_pos_ratio=NEG_POS_RATIO
    )
    
    print("\n✅ Pipeline completado exitosamente!")
    print(f"📂 Archivos generados en: {OUTPUT_DIR}")
    print("\n🎯 Próximos pasos:")
    print("   1. Cargar datos: X_train, y_train, _ = dataset.get_feature_labels('train')")
    print("   2. Entrenar modelo: rf.fit(X_train, y_train)")
    print("   3. Evaluar: rf.score(X_test, y_test)")
    
    # Ejemplo de carga de datos
    print("\n" + "="*60)
    print("📥 EJEMPLO DE CARGA DE DATOS")
    print("="*60)
    
    try:
        # Cargar split de entrenamiento (normalizado)
        X_train, y_train, meta_train = dataset.get_feature_labels("train", normalized=True)
        
        print(f"\n✅ Datos de entrenamiento cargados:")
        print(f"   • Shape de X: {X_train.shape}")
        print(f"   • Shape de y: {y_train.shape}")
        print(f"   • Columnas de X: {list(X_train.columns)}")
        print(f"\n   Distribución de clases:")
        print(f"   • No-fuego (0): {(y_train == 0).sum():,}")
        print(f"   • Fuego (1): {(y_train == 1).sum():,}")
        
    except Exception as e:
        print(f"\n⚠️  Error al cargar datos: {e}")
        print("   Asegúrate de haber ejecutado el pipeline completo primero.")