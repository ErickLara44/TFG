import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ================================
# CONFIGURACIÓN INICIAL
# ================================

# Ruta al DataCube NetCDF
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"

# Abrir el DataCube con xarray
ds = xr.open_dataset(datacube_path)
import pandas as pd

# Extraer variables y crear tabla resumen
variables_info = []

for var_name, var in ds.data_vars.items():
    variables_info.append({
        "Variable": var_name,
        "Dimensiones": list(var.dims),
        "Shape": list(var.shape),
        "Tipo": str(var.dtype),
        "Unidades": var.attrs.get("units", "sin unidades"),
        "Descripción": var.attrs.get("long_name", "sin descripción")
    })

df_vars = pd.DataFrame(variables_info)
pd.set_option("display.max_rows", None)  # mostrar todas las filas
print(df_vars)

