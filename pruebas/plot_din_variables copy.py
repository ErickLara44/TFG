import xarray as xr
import matplotlib.pyplot as plt

# ================================
# CONFIGURACIÓN
# ================================
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"
ds = xr.open_dataset(datacube_path)

var = "is_holiday"
fecha = "2022-05-02"  # festivo SOLO en Comunidad de Madrid

region_spain = dict(
     x=slice(2_674_734, 3_861_734),  # rango completo X
    y=slice(2_492_195, 1_573_195)
)

# ================================
# EXTRACCIÓN DE DATOS
# ================================
v = ds[var].sel(**region_spain, time=fecha)

# ================================
# VISUALIZACIÓN
# ================================
plt.figure(figsize=(10, 8))
v.plot(
    cmap="coolwarm",
    vmin=0, vmax=1,
    cbar_kwargs={"ticks": [0, 1], "label": "Festivo (1) / Laboral (0)"}
)
plt.title(f"Festivo en  España - {fecha}")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(False)
plt.tight_layout()
plt.show()

ds.close()