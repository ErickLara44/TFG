import xarray as xr
import pandas as pd
import numpy as np
import os
import gc
import psutil
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt


class SpainCubeFireDataset:
    """
    Genera dataset tabular comprimido (Parquet) por celda y día.
    Filtra por CCAA y divide en train/val/test automáticamente.
    """

    def __init__(self, datacube_path,
                 output_dir="data/processed",
                 start_year=2024, end_year=2024,
                 ccaa_target=None,
                 chunk_size_days=1,
                 neg_pos_ratio=1.5):
        
        # ============================================================
        # 🔥 CONFIGURACIÓN PRINCIPAL
        # ============================================================
        if ccaa_target is None:
            # Galicia (12), Castilla y León (7), Extremadura (11),
            # Andalucía (1), Asturias (3)
            ccaa_target = [12, 7, 11, 1, 3]

        self.datacube_path = datacube_path
        self.output_dir = output_dir
        self.start_year = start_year
        self.end_year = end_year
        self.ccaa_target = ccaa_target
        self.chunk_size_days = chunk_size_days
        self.neg_pos_ratio = neg_pos_ratio

        os.makedirs(output_dir, exist_ok=True)

        # Archivos finales
        self.parquet_dir = os.path.join(output_dir, "by_year")
        os.makedirs(self.parquet_dir, exist_ok=True)

        self.train_file = os.path.join(output_dir, "train.parquet")
        self.val_file   = os.path.join(output_dir, "val.parquet")
        self.test_file  = os.path.join(output_dir, "test.parquet")
        self.scaler_path = os.path.join(output_dir, "scaler.pkl")

        self.feature_vars = self._get_feature_variables()

        # ============================================================
        # 🌍 MAPA DE CCAA Y LOG INICIAL
        # ============================================================
        self.CCAA_NAMES = {
            1: "Andalucía", 3: "Asturias", 6: "Cantabria", 7: "Castilla y León",
            8: "Castilla-La Mancha", 10: "Comunidad Valenciana",
            11: "Extremadura", 12: "Galicia", 16: "País Vasco"
        }

        print("\n🌍 Comunidades seleccionadas para el dataset:")
        for code in self.ccaa_target:
            name = self.CCAA_NAMES.get(code, f"Desconocida ({code})")
            print(f"   • {name} ({code})")
        print("=============================================================\n")

    # ----------------------------------------------------------------------
    def _get_feature_variables(self):
        return [
            'elevation_mean', 'slope_mean',
            'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion',
            'CLC_2018_agricultural_proportion',
            'dist_to_roads_mean', 'popdens_2018', 'is_waterbody',
            't2m_mean', 'RH_min', 'wind_speed_mean', 'wind_direction_mean',
            'total_precipitation_mean', 'NDVI', 'SWI_010', 'FWI', 'LST',
            'is_near_fire','x_coordinate','y_coordinate'
        ]

    # ----------------------------------------------------------------------
    def extract_raw_data(self):
        """
        ✅ CORREGIDO: Acumula DataFrames en memoria y guarda al final.
        """
        print("🧩 Extrayendo datos del datacube...")
        ds = xr.open_dataset(self.datacube_path)

        start_date = f"{self.start_year}-01-01"
        end_date   = f"{self.end_year}-12-31"
        ds = ds.sel(time=slice(start_date, end_date))

        # Filtrar solo España
        if "is_spain" in ds:
            ds = ds.where(ds["is_spain"] == 1, drop=True)
            print("✅ Filtrado aplicado: solo celdas con is_spain=1")

        # Variables necesarias
        vars_needed = self.feature_vars + ['is_fire', 'AutonomousCommunities']
        vars_available = [v for v in vars_needed if v in ds.data_vars or v in ds.coords]
        ds = ds[vars_available]

        time_index = ds.time.values
        total_days = len(time_index)
        print(f"📅 Días encontrados: {total_days}")

        print(f"\n🧮 Iniciando procesamiento...")
        print(f" - Chunk temporal: {self.chunk_size_days} día(s)")
        print(f" - Total días a procesar: {total_days}")
        print(f" - Comunidades objetivo: {[self.CCAA_NAMES.get(c, c) for c in self.ccaa_target]}")
        print(f" - Variables: {vars_available}\n")

        # ✅ FIX 1: Usar diccionario para acumular DataFrames por año
        year_buffers = {}
        processed_total = 0
        avg_rows_per_day = []

        # Progreso detallado
        for i, start in enumerate(range(0, total_days, self.chunk_size_days), 1):
            end = min(start + self.chunk_size_days, total_days)
            ds_chunk = ds.isel(time=slice(start, end))
            df_chunk = ds_chunk.to_dataframe().reset_index()

            # Filtrar comunidades
            if "AutonomousCommunities" in df_chunk.columns:
                df_chunk = df_chunk[df_chunk["AutonomousCommunities"].isin(self.ccaa_target)]

            if len(df_chunk) == 0:
                continue

            # Añadir año
            df_chunk["year"] = pd.to_datetime(df_chunk["time"]).dt.year

            # ✅ Acumular en buffers en lugar de append
            for year, df_y in df_chunk.groupby("year"):
                if year not in year_buffers:
                    year_buffers[year] = []
                year_buffers[year].append(df_y)

            # Estadísticas
            processed_total += len(df_chunk)
            avg_rows_per_day.append(len(df_chunk))
            avg = np.mean(avg_rows_per_day)

            if i % 10 == 0 or i == 1 or end == total_days:
                print(f"📆 Día {end}/{total_days} "
                      f"| Filas procesadas: {processed_total:,} "
                      f"| Promedio: {int(avg):,}/día "
                      f"| Progreso: {end/total_days*100:.1f}%")

            del ds_chunk, df_chunk
            gc.collect()

        # ✅ Guardar buffers acumulados
        print("\n💾 Guardando archivos Parquet por año...")
        for year, dfs in year_buffers.items():
            combined = pd.concat(dfs, ignore_index=True)
            parquet_path = os.path.join(self.parquet_dir, f"raw_{year}.parquet")
            combined.to_parquet(parquet_path, compression="snappy")
            print(f"   • {year}: {len(combined):,} filas → {parquet_path}")

        ds.close()
        print(f"\n✅ Extracción completada. Total filas procesadas: {processed_total:,}")
        print(f"📁 Archivos guardados en: {self.parquet_dir}")

    # ----------------------------------------------------------------------
    def balance_and_split(self):
        """
        ✅ CORREGIDO: Divide PRIMERO por año, balancea DESPUÉS cada split.
        """
        print("\n📂 Cargando datos por año...")
        files = [os.path.join(self.parquet_dir, f) for f in os.listdir(self.parquet_dir) 
                 if f.endswith(".parquet")]
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["year"] = pd.to_datetime(df["time"]).dt.year

        print(f"📊 Total filas cargadas: {len(df):,}")
        print(f"📊 Distribución por año:")
        for year in sorted(df['year'].unique()):
            n_total = len(df[df['year'] == year])
            n_fire = (df[df['year'] == year]['is_fire'] == 1).sum()
            print(f"   • {year}: {n_total:,} filas ({n_fire:,} fuegos)")

        # ✅ FIX 2: Dividir PRIMERO por año
        train = df[df["year"] <= 2022].copy()
        val   = df[df["year"] == 2023].copy()
        test  = df[df["year"] == 2024].copy()

        print(f"\n🔪 División temporal:")
        print(f"   • Train (≤2022): {len(train):,} filas")
        print(f"   • Val (2023):    {len(val):,} filas")
        print(f"   • Test (2024):   {len(test):,} filas")

        # ✅ Balancear CADA split independientemente
        def balance_split(df_split, ratio, split_name):
            fire = df_split[df_split["is_fire"] == 1]
            nofire = df_split[df_split["is_fire"] == 0]
            
            n_nofire_needed = int(len(fire) * ratio)
            n_nofire_actual = min(n_nofire_needed, len(nofire))
            
            nofire_sampled = nofire.sample(n=n_nofire_actual, random_state=42)
            balanced = pd.concat([fire, nofire_sampled]).sample(frac=1, random_state=42)
            
            print(f"\n   {split_name}:")
            print(f"      - Fuegos: {len(fire):,}")
            print(f"      - No fuegos (original): {len(nofire):,}")
            print(f"      - No fuegos (sampled): {n_nofire_actual:,}")
            print(f"      - Ratio final: {n_nofire_actual/len(fire):.2f}")
            print(f"      - Total balanceado: {len(balanced):,}")
            
            return balanced

        print(f"\n⚖️  Balanceando cada split (ratio objetivo: {self.neg_pos_ratio})...")
        train_balanced = balance_split(train, self.neg_pos_ratio, "TRAIN")
        val_balanced   = balance_split(val, self.neg_pos_ratio, "VAL")
        test_balanced  = balance_split(test, self.neg_pos_ratio, "TEST")

        # Guardar
        train_balanced.to_parquet(self.train_file, compression="snappy")
        val_balanced.to_parquet(self.val_file, compression="snappy")
        test_balanced.to_parquet(self.test_file, compression="snappy")

        print(f"\n✅ Archivos guardados:")
        print(f"   • {self.train_file}")
        print(f"   • {self.val_file}")
        print(f"   • {self.test_file}")

        # 🔍 Visualización de distribución (ANTES del balanceo)
        fire_total = df[df["is_fire"] == 1]
        nofire_total = df[df["is_fire"] == 0]
        
        plt.figure(figsize=(8, 5))
        plt.bar(["Fuego", "No Fuego"], [len(fire_total), len(nofire_total)], 
                color=["#ff4444", "#888888"], alpha=0.7, edgecolor='black')
        plt.title("Distribución de clases (ANTES del balanceo)", fontsize=14, fontweight='bold')
        plt.ylabel("Número de instancias", fontsize=12)
        plt.yscale('log')  # Escala log para ver mejor la diferencia
        for i, (label, count) in enumerate(zip(["Fuego", "No Fuego"], [len(fire_total), len(nofire_total)])):
            plt.text(i, count, f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "class_distribution.png"), dpi=150)
        plt.show()

    # ----------------------------------------------------------------------
    def normalize_data(self):
        """
        ✅ CORREGIDO: Solo normaliza variables continuas (excluye binarias).
        """
        print("\n📏 Normalizando features...")
        df_train = pd.read_parquet(self.train_file)
        
        # ✅ FIX 3: Excluir variables binarias/categóricas de la normalización
        binary_vars = ['is_waterbody', 'is_near_fire']  # Variables que son 0/1
        
        # Solo normalizar features numéricas continuas
        cols_to_normalize = [c for c in self.feature_vars 
                            if c in df_train.columns and c not in binary_vars]
        
        print(f"   • Variables a normalizar ({len(cols_to_normalize)}): {cols_to_normalize}")
        print(f"   • Variables binarias (sin normalizar): {binary_vars}")

        # Fit scaler solo en train
        scaler = StandardScaler()
        scaler.fit(df_train[cols_to_normalize])
        
        # Guardar scaler
        with open(self.scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"   • Scaler guardado en: {self.scaler_path}")

        # Aplicar normalización a cada split
        for name, path in [("train", self.train_file), 
                          ("val", self.val_file), 
                          ("test", self.test_file)]:
            df = pd.read_parquet(path)
            
            # Normalizar solo las columnas continuas
            df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
            
            # Guardar versión normalizada
            normalized_path = path.replace(".parquet", "_normalized.parquet")
            df.to_parquet(normalized_path, compression="snappy")
            print(f"   ✅ {name.upper()} normalizado → {normalized_path}")

    # ----------------------------------------------------------------------
    def get_feature_labels(self, split="train", normalized=True):
        """Carga X, y y metadatos desde Parquet"""
        suffix = "_normalized.parquet" if normalized else ".parquet"
        
        if split == "train":
            path = self.train_file.replace(".parquet", suffix)
        elif split == "val":
            path = self.val_file.replace(".parquet", suffix)
        elif split == "test":
            path = self.test_file.replace(".parquet", suffix)
        else:
            raise ValueError("split debe ser 'train', 'val' o 'test'")

        df = pd.read_parquet(path)
        
        # ✅ Verificar que todas las features existen
        missing_features = [f for f in self.feature_vars if f not in df.columns]
        if missing_features:
            print(f"⚠️  Advertencia: Features faltantes en {split}: {missing_features}")
        
        available_features = [f for f in self.feature_vars if f in df.columns]
        
        X = df[available_features].copy()
        y = df["is_fire"].copy()
        metadata = df[["x", "y", "time"]].copy()
        
        return X, y, metadata
    
    # ----------------------------------------------------------------------
    def print_dataset_summary(self):
        """Imprime resumen completo del dataset generado"""
        print("\n" + "="*60)
        print("📊 RESUMEN DEL DATASET GENERADO")
        print("="*60)
        
        for split_name in ["train", "val", "test"]:
            X, y, meta = self.get_feature_labels(split_name, normalized=False)
            
            n_fire = (y == 1).sum()
            n_nofire = (y == 0).sum()
            ratio = n_nofire / n_fire if n_fire > 0 else 0
            
            print(f"\n{split_name.upper()}:")
            print(f"  • Total samples: {len(y):,}")
            print(f"  • Fuegos: {n_fire:,} ({n_fire/len(y)*100:.1f}%)")
            print(f"  • No fuegos: {n_nofire:,} ({n_nofire/len(y)*100:.1f}%)")
            print(f"  • Ratio no-fuego/fuego: {ratio:.2f}")
            print(f"  • Rango temporal: {meta['time'].min()} → {meta['time'].max()}")
        
        print("\n" + "="*60 + "\n")