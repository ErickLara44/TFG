import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# ---------- Abrir dataset ----------
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"
ds = xr.open_dataset(datacube_path)

# ---------- 1. Feature espacial: Elevación ----------
plt.figure(figsize=(8,6))
elevation = ds['CLC_2018_9']
elevation.plot(cmap="Reds", cbar_kwargs={'label': 'LANDCOVER (m)'})
plt.title("Elevación media en España")
plt.show()

# ---------- 2. Feature temporal: Temperatura en una fecha ----------
plt.figure(figsize=(8,6))
temp_20220715 = ds['t2m_mean'].sel(time="2022-06-18")
temp_20220715.plot(cmap="coolwarm", cbar_kwargs={'label': 'Temperature (°C)'})
plt.title("Temperatura media - 18 Junio 2022")
plt.show()

# ---------- 3. Máscara de España ----------
plt.figure(figsize=(8,6))
mask_spain = ds['is_spain']
data_masked = temp_20220715.where(mask_spain == 1)  # Aplicar máscara
data_masked.plot(cmap="coolwarm", cbar_kwargs={'label': 'Temperature (°C)'})
plt.title("Temperatura media en España (15 Julio 2022)")
plt.show()

# ---------- 4. Visualización de incendios ----------
plt.figure(figsize=(8,6))
fire = ds["is_fire"].sel(time="2022-06-18")  # Ojo: usar formato YYYY-MM-DD
fire.where(ds["is_spain"] == 1).plot(cmap="Reds", cbar_kwargs={'label': 'Fire (1 = fire)'})
plt.title("Incendios detectados - 8 Febrero 2024")
plt.show()


