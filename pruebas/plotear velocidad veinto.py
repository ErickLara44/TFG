import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# ================================
# CONFIGURACIÓN INICIAL
# ================================
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"
ds = xr.open_dataset(datacube_path)

# Variables de viento
var_speed = "wind_speed_mean"
var_dir = "wind_direction_mean"

# Región: toda España
region_spain = dict(
    x=slice(2_674_734, 3_861_734),
    y=slice(2_492_195, 1_573_195)
)

# Día a visualizar
fecha = "2022-07-01"

# ================================
# EXTRACCIÓN DE DATOS
# ================================
# Seleccionar datos de velocidad y dirección para ese día y región
speed = ds[var_speed].sel(**region_spain, time=fecha)
direction = ds[var_dir].sel(**region_spain, time=fecha)

# ================================
# CÁLCULO DE COMPONENTES U y V
# ================================
# Convertir dirección (grados meteorológicos) → radianes
# Dirección meteorológica = de dónde viene el viento
theta = np.deg2rad(direction)

# Calcular componentes del viento (hacia dónde sopla)
U = -speed * np.sin(theta)  # componente este-oeste
V = -speed * np.cos(theta)  # componente norte-sur

# ================================
# VISUALIZACIÓN
# ================================
# Reducir resolución para graficar menos flechas (más legible)
step = 10  # cambiar a 5 o 20 según densidad deseada
X = ds["x"].values
Y = ds["y"].values

# Crear figura
plt.figure(figsize=(10, 8))

# Fondo coloreado: magnitud del viento
plt.pcolormesh(X, Y, speed, cmap="Blues", shading="auto")
plt.colorbar(label="Velocidad del viento (m/s)")

# Superponer las flechas del viento
plt.quiver(
    X[::step],
    Y[::step],
    U.values[::step, ::step],
    V.values[::step, ::step],
    color="white",
    scale=300,       # controla longitud de flechas
    width=0.002,     # grosor de flechas
    headwidth=3
)

# Etiquetas y título
plt.title(f"Campo de viento medio - {fecha}")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(False)
plt.tight_layout()
plt.show()

# ================================
# LIMPIEZA
# ================================
ds.close()