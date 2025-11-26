import xarray as xr
import matplotlib.pyplot as plt

# ================================
# CONFIGURACIÓN INICIAL
# ================================
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"
ds = xr.open_dataset(datacube_path)
print("Variables disponibles:", list(ds.data_vars.keys()))

# Variable estática a visualizar
var = "CLC_2018_agricultural_proportion"

# ================================
# REGIÓN: TODA ESPAÑA
# ================================
region_spain = dict(
    x=slice(2_674_734, 3_861_734),  # rango completo X
    y=slice(2_492_195, 1_573_195)   # rango completo Y (de norte → sur)
)

# Extraer directamente (sin 'time', porque elevation_mean es estática)
v = ds[var].sel(**region_spain).squeeze()
print(v)

# ================================
# CÁLCULO ESTADÍSTICO
# ================================
mean_value = float(v.mean().values)
std_value = float(v.std().values)
max_value = float(v.max().values)
min_value = float(v.min().values)

print("--- CLC_2018_agricultural_proportion---")
print(f"Media: {mean_value:.2f} m")
print(f"Desviación estándar: {std_value:.2f} m")
print(f"Máximo: {max_value:.2f} m")
print(f"Mínimo: {min_value:.2f} m")

# ================================
# VISUALIZACIÓN
# ================================
plt.figure(figsize=(10, 8))
v.plot(cmap="terrain")
plt.title(f"CLC_2018_agricultural_proportion = {mean_value:.1f} m")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.show()

# ================================
# LIMPIEZA
# ================================
ds.close()