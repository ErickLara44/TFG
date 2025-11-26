import xarray as xr

# Abrir datacube
datacube_path = "/Users/erickmollinedolara/Erick/Uni/TFG/Cursor/iberfire/data/IberFire.nc"
ds = xr.open_dataset(datacube_path)

print("=== INFORMACIÓN DEL DATACUBE ===")
print(f"Dimensiones: {ds.dims}")
print(f"Coordenadas: {list(ds.coords.keys())}")
print(f"Variables de datos: {list(ds.data_vars.keys())}")
print(f"Total de variables: {len(ds.data_vars)}")

print("\n=== VARIABLES QUE ESTÁS BUSCANDO ===")
variables_buscadas = [
    'elevation_mean', 'slope_mean',
    'CLC_2018_forest_proportion', 'CLC_2018_scrub_proportion',
    'CLC_2018_agricultural_proportion',
    'dist_to_roads_mean', 'popdens_2018', 'is_waterbody',
    't2m_mean', 'RH_min', 'wind_speed_mean', 'wind_direction_mean',
    'total_precipitation_mean', 'NDVI', 'SWI_010', 'FWI', 'LST',
    'is_near_fire', 'x_coordinate', 'y_coordinate', 'is_holiday'
]

for var in variables_buscadas:
    if var in ds.data_vars:
        print(f"✅ {var} - ENCONTRADA")
    elif var in ds.coords:
        print(f"📍 {var} - ES COORDENADA")
    else:
        print(f"❌ {var} - NO ENCONTRADA")

print("\n=== PRIMERAS 10 VARIABLES DISPONIBLES ===")
for i, var in enumerate(list(ds.data_vars.keys())[:10]):
    print(f"{i+1}. {var}")

ds.close()



