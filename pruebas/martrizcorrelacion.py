import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# CONFIGURACIÓN
# ==========================
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"

# Solo variables estáticas
variables = [
    'elevation_mean', 'slope_mean',
    'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion',
    'CLC_2018_agricultural_proportion',
    'dist_to_roads_mean', 'popdens_2018', 'is_waterbody',
    't2m_mean', 'RH_min', 'wind_speed_mean', 'wind_direction_mean',
    'total_precipitation_mean', 'NDVI', 'SWI_010', 'FWI', 'LST',
    'is_near_fire', 'x_coordinate', 'y_coordinate', 'is_holiday','is_fire'
]
# ==========================
# 1️⃣ CARGA (lazy, sin DataFrame)
# ==========================
ds = xr.open_dataset(datacube_path, chunks={"x": 100, "y": 100})
print("[INFO] Dataset abierto (lazy).")

# Filtrar variables que existan realmente
vars_missing = set(variables) - set(ds.data_vars)
if vars_missing:
    print(f"[WARN] Variables no encontradas: {vars_missing}")
    variables = [v for v in variables if v in ds.data_vars]

ds = ds[variables]

# ==========================
# 2️⃣ NORMALIZACIÓN Y REDUCCIÓN ESPACIAL
# ==========================
# Quitamos valores NaN, promediamos sobre una cuadrícula más gruesa (reduce RAM)
ds_small = ds.coarsen(x=10, y=10, boundary="trim").mean()
print("[INFO] Dataset reducido con coarsen:", ds_small.dims)

# ==========================
# 3️⃣ CÁLCULO DE CORRELACIÓN EN XARRAY (sin pasar a pandas)
# ==========================
# Convertimos en array (var, y, x)
arr = ds_small.to_array(dim="variable")

# Calculamos correlación espacial entre variables
corr_matrix = xr.corr(arr, arr, dim=("x", "y"))

# Convertimos a DataFrame para graficar (esto ya es pequeño)
corr_df = corr_matrix.to_pandas()

# ==========================
# 4️⃣ VISUALIZACIÓN
# ==========================
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_df, cmap="coolwarm", vmin=-1, vmax=1,
    square=True, linewidths=0.1, cbar_kws={"shrink": 0.7}
)
plt.title("Matriz de correlaciones (solo variables estáticas, coarsened)", fontsize=13)
plt.tight_layout()
plt.show()

ds.close()

# ✅ VARIABLES FIJAS PARA TODOS LOS PROCESOS
