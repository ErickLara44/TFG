import xarray as xr
import pandas as pd

# Ruta a tu datacube
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"

# Abrir dataset
ds = xr.open_dataset(datacube_path)

print("==============================================")
print("📦 INFORMACIÓN GENERAL DEL DATACUBE")
print("==============================================")
print(ds)

print("\n==============================================")
print("🧭 DIMENSIONES")
print("==============================================")
for dim, size in ds.dims.items():
    print(f"  • {dim}: {size}")

print("\n==============================================")
print("🌡️ VARIABLES DISPONIBLES")
print("==============================================")
var_info = []
for var in ds.data_vars:
    v = ds[var]
    var_info.append({
        "Variable": var,
        "Dims": ", ".join(list(v.dims)),
        "Shape": str(list(v.shape)),
        "Tipo": str(v.dtype),
        "Descripción": v.attrs.get("long_name", ""),
        "Unidad": v.attrs.get("units", "")
    })

df_vars = pd.DataFrame(var_info)
pd.set_option('display.max_rows', None)  # Mostrar todas
print(df_vars)

print("\n==============================================")
print("🧭 COORDENADAS")
print("==============================================")
for coord in ds.coords:
    print(f"  • {coord}: {list(ds[coord].shape)}")

# Si quieres ver los rangos espaciales y temporales:
print("\n==============================================")
print("📅 RANGOS TEMPORALES Y ESPACIALES")
print("==============================================")
if "time" in ds.coords:
    print(f"  • Tiempo: {str(ds.time.values[0])} → {str(ds.time.values[-1])}")
if "x" in ds.coords and "y" in ds.coords:
    print(f"  • X: {float(ds.x.min())} → {float(ds.x.max())}")
    print(f"  • Y: {float(ds.y.min())} → {float(ds.y.max())}")

# Cerrar dataset
ds.close()