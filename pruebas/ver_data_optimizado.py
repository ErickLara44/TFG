import xarray as xr
import pandas as pd
import numpy as np
import random
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------
# CONFIGURACIÓN
# -------------------------
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"
# -------------------------
# VARIABLES SELECCIONADAS
# -------------------------
selected_variables = [
    # 1️⃣ Coordenadas y ubicación
    'x_coordinate', 'y_coordinate',
    'is_spain', 'is_sea', 'is_waterbody',
    
    # 2️⃣ Historia de incendios
    'is_fire', 'is_near_fire',
    
    # 3️⃣ Uso del suelo (CLC 2006)
    'CLC_2006_1','CLC_2006_44','CLC_2006_urban_fabric_proportion','CLC_2006_industrial_proportion',
    'CLC_2006_mine_proportion','CLC_2006_artificial_vegetation_proportion','CLC_2006_arable_land_proportion',
    'CLC_2006_permanent_crops_proportion','CLC_2006_heterogeneous_agriculture_proportion',
    'CLC_2006_forest_proportion','CLC_2006_scrub_proportion','CLC_2006_open_space_proportion',
    'CLC_2006_inland_wetlands_proportion','CLC_2006_maritime_wetlands_proportion',
    'CLC_2006_inland_waters_proportion','CLC_2006_marine_waters_proportion',
    'CLC_2006_artificial_proportion','CLC_2006_agricultural_proportion',
    'CLC_2006_forest_and_semi_natural_proportion','CLC_2006_wetlands_proportion',
    'CLC_2006_waterbody_proportion',
    
    # 4️⃣ Topografía y terreno
    'elevation_mean','elevation_stdev','slope_mean','slope_stdev',
    'roughness_mean','roughness_stdev',
    'aspect_1','aspect_2','aspect_3','aspect_4','aspect_5','aspect_6','aspect_7','aspect_8','aspect_NODATA',
    
    # 5️⃣ Proximidad a infraestructuras
    'dist_to_roads_mean','dist_to_roads_stdev','dist_to_waterways_mean','dist_to_waterways_stdev',
    'dist_to_railways_mean','dist_to_railways_stdev',
    
    # 6️⃣ Indicadores humanos y legales
    'is_holiday','is_natura2000','popdens_2008','popdens_2020',
    
    # 7️⃣ Variables meteorológicas y ambientales
    't2m_mean','t2m_min','t2m_max','t2m_range',
    'RH_mean','RH_min','RH_max','RH_range',
    'surface_pressure_mean','surface_pressure_min','surface_pressure_max','surface_pressure_range',
    'total_precipitation_mean',
    'wind_speed_mean','wind_speed_max','wind_direction_mean','wind_direction_at_max_speed',
    'FAPAR','LAI','LST','NDVI',
    'SWI_001','SWI_005','SWI_010','SWI_020'
]
years_to_use = [2022]
negatives_per_positive = 5
chunk_size_time = 15  # días por chunk
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# -------------------------
# 1️⃣ Abrir datacube y filtrar
# -------------------------
print("[INFO] Abriendo datacube...")
ds = xr.open_dataset(datacube_path)
ds = ds[selected_variables]
ds = ds.sel(time=ds["time.year"].isin(years_to_use))
print(f"[INFO] Datacube filtrado por variables y años {years_to_use}")

# -------------------------
# 2️⃣ Generación de instancias chunked
# -------------------------
all_chunks = []
time_index = ds.time.values
print(f"[INFO] Número de días a procesar: {len(time_index)}")

for start in range(0, len(time_index), chunk_size_time):
    end = min(start + chunk_size_time, len(time_index))
    ds_chunk = ds.sel(time=time_index[start:end])
    df_chunk = ds_chunk.to_dataframe().reset_index()
    
    # Positivas
    df_fire = df_chunk[df_chunk["is_fire"] == 1]
    
    # Negativas (muestreo)
    n_neg = len(df_fire) * negatives_per_positive
    df_nofire_pool = df_chunk[df_chunk["is_fire"] == 0]
    if len(df_nofire_pool) > 0:
        df_nofire_sample = df_nofire_pool.sample(n=n_neg, replace=True, random_state=random_seed)
        df_nofire_sample["is_fire"] = 0
        df_chunk_final = pd.concat([df_fire, df_nofire_sample], ignore_index=True)
    else:
        df_chunk_final = df_fire.copy()
    
    all_chunks.append(df_chunk_final)
    print(f"[INFO] Chunk {start}-{end} procesado, filas acumuladas: {sum(len(c) for c in all_chunks)}")

# -------------------------
# 3️⃣ Concatenar todos los chunks
# -------------------------
final_dataset = pd.concat(all_chunks, ignore_index=True)
del all_chunks
print(f"[INFO] Dataset concatenado final: {len(final_dataset)} filas")

# -------------------------
# 4️⃣ LIMPIEZA DE DATOS
# -------------------------
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

numeric_columns = [col for col in final_dataset.columns if col != 'is_fire']
for col in numeric_columns:
    final_dataset[col] = pd.to_numeric(final_dataset[col], errors='coerce')

# CLC proporciones
clc_columns = [c for c in final_dataset.columns if "CLC" in c]
if clc_columns:
    final_dataset['clc_sum'] = final_dataset[clc_columns].sum(axis=1)
    for col in clc_columns:
        final_dataset[col] = final_dataset[col] / final_dataset['clc_sum']
    final_dataset.drop(columns=['clc_sum'], inplace=True)

# Aspect proporciones
aspect_columns = [c for c in final_dataset.columns if "aspect" in c]
if aspect_columns:
    final_dataset['aspect_sum'] = final_dataset[aspect_columns].sum(axis=1)
    for col in aspect_columns:
        final_dataset[col] = final_dataset[col] / final_dataset['aspect_sum']
    final_dataset.drop(columns=['aspect_sum'], inplace=True)

# Natura2000 proporciones
if "natura2000_1" in final_dataset.columns and "natura2000_NODATA" in final_dataset.columns:
    total = final_dataset["natura2000_1"] + final_dataset["natura2000_NODATA"]
    final_dataset["natura2000_1"] = final_dataset["natura2000_1"] / total
    final_dataset.drop(columns=["natura2000_NODATA"], inplace=True)

warnings.simplefilter(action='default', category=pd.errors.SettingWithCopyWarning)

# -------------------------
# 5️⃣ Guardar dataset final
# -------------------------
final_csv_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/FinalDataset.csv"
final_dataset.to_csv(final_csv_path, index=False)
print(f"[INFO] Dataset final limpio guardado en: {final_csv_path}")

# -------------------------
# 6️⃣ Proporción de fuego/no-fuego
# -------------------------@
fire_counts = Counter(final_dataset["is_fire"])
print(f"[INFO] Proporción final de instancias:")
print(fire_counts)

# -------------------------
# 7️⃣ MATRIZ DE CORRELACIÓN
# -------------------------
non_numeric_cols = ['initialdate', 'finaldate']  # si existen
df_numeric = final_dataset.drop(columns=[c for c in non_numeric_cols if c in final_dataset.columns])
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(20, 20))
sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    fmt='.2f',
    cbar=True,
    square=True,
    linewidths=0.1,
    cbar_kws={'shrink': 0.5}
)
plt.title('Matriz de correlaciones', fontsize=18)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()