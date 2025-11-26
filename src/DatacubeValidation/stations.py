import geopandas as gpd
import pandas as pd
import os
import numpy as np
from src.config import AEMET_STATIONS_FILE, VALIDATION_DATASETS_FOLDER



def dms_str_to_decimal(dms_str):
    """
    Convert a coordinate in DMS, Degrees Minutes Seconds format 
    (e.g., "082219W" or "1082219E") to decimal degrees.
    
    The function expects the last character to be a hemisphere indicator:
      - For latitudes: 'N' or 'S'
      - For longitudes: 'E' or 'W'
    
    The numeric part (degrees, minutes, seconds) is determined by:
      - If len(numeric part) == 6, degrees are the first 2 digits.
      - If len(numeric part) == 7, degrees are the first 3 digits.
    """
    dms_str = dms_str.strip()
    if not dms_str:
        return None  # Handle empty strings if needed.
        
    # Get the hemisphere letter (last character)
    hemisphere = dms_str[-1].upper()
    # Get the numeric part of the string
    num_part = dms_str[:-1]
    
    # Determine the number of digits used for degrees
    if len(num_part) == 6:
        # Format: DDMMSS
        degrees = int(num_part[:2])
        minutes = int(num_part[2:4])
        seconds = int(num_part[4:])
    elif len(num_part) == 7:
        # Format: DDDMMSS
        degrees = int(num_part[:3])
        minutes = int(num_part[3:5])
        seconds = int(num_part[5:])
    else:
        raise ValueError(f"Unexpected format for DMS string: {dms_str}")
    
    # Convert to decimal degrees
    decimal = degrees + minutes / 60 + seconds / 3600
    
    # Negative for South and West hemispheres
    if hemisphere in ['S', 'W']:
        decimal *= -1
    
    return decimal


def get_all_aemet_stations_geodataframe():
    """
    Load all AEMET stations from a CSV file, convert DMS coordinates to decimal degrees,
    delete Canary Islands, Ceuta and Melilla stations, and return a GeoDataFrame with
    the station data on the EPSG:3035 coordinate reference system.
    """
    stations = pd.read_csv(AEMET_STATIONS_FILE, encoding='latin-1', sep = ";", header=None)
    stations.columns = ["INDICATIVO", "INDSINOP", "NOMBRE", "PROVINCIA", "LATITUD", "LONGITUD", "ALTURA", "FECHA_INICIO", "FECHA_FIN"]

    stations['LATITUD'] = stations['LATITUD'].apply(dms_str_to_decimal)
    stations['LONGITUD'] = stations['LONGITUD'].apply(dms_str_to_decimal)
    
    gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.LONGITUD, stations.LATITUD),
        crs="EPSG:4326"
    )

    gdf_proj = gdf.to_crs("EPSG:3035")
    gdf_proj["X_COORDINATE"] = gdf_proj.geometry.x
    gdf_proj["Y_COORDINATE"] = gdf_proj.geometry.y

    return gdf_proj


def get_invalid_aemet_stations_ids():
    all_stations = get_all_aemet_stations_geodataframe()
    # Filter out stations outside the datacube area of interest
    invalid_stations = all_stations[all_stations["PROVINCIA"].isin(["LAS PALMAS", 
                                                                "STA. CRUZ DE TENERIFE",
                                                                "SANTA CRUZ DE TENERIFE",
                                                                "CEUTA", "MELILLA"])]["INDICATIVO"].tolist()
    invalid_stations += all_stations[all_stations["NOMBRE"].isin(["ALBORÁN"])]["INDICATIVO"].tolist()
    # Filter out stations outside the datacube period
    for _, row in all_stations.iterrows():
        fecha_inicio = row['FECHA_INICIO']
        fecha_fin = row["FECHA_FIN"]

        fecha_inicio = pd.to_datetime(fecha_inicio)
        fecha_fin = pd.to_datetime(fecha_fin)

        if fecha_fin <= pd.Timestamp("2007-12-01") or fecha_inicio >= pd.Timestamp("2024-12-31"):
            invalid_stations.append(row['INDICATIVO'])
    # Filter out stations with errors in data retrieval
    for _, row in all_stations.iterrows():
        try: get_aemet_station_data_from_id(row['INDICATIVO'], warnings=False)
        except:
            invalid_stations.append(row['INDICATIVO'])
    return invalid_stations

def get_valid_aemet_stations_geodataframe():
    all_stations = get_all_aemet_stations_geodataframe()
    return all_stations[~all_stations["INDICATIVO"].isin(get_invalid_aemet_stations_ids())]

def get_aemet_station_path_from_id(station_id):
    """
    Get the path of the AEMET station data file by its ID.
    
    Args:
        station_id (str): The station ID to search for.
        
    Returns:
        str: The path to the station data file, or None if not found.
    """
    files = [f for f in os.listdir(VALIDATION_DATASETS_FOLDER) if f.endswith(".csv")]
    station_ids = [f.split("-")[0] for f in files]

    if station_id in station_ids:
        station_file = [f for f in files if f.startswith(station_id)][0]
        station_path = os.path.join(VALIDATION_DATASETS_FOLDER, station_file)
        return station_path
    return None

def get_aemet_station_data_from_id(station_id, warnings = True):
    """
    Get AEMET station data by its ID.
    
    Args:
        station_id (str): The station ID to search for.
        
    Returns:
        dataframe: A pandas DataFrame containing the station data
        from 2007-12-01 to 2024-12-31, or None if not found.
    """
    stations = get_all_aemet_stations_geodataframe()
    station_data = stations[stations['INDICATIVO'] == station_id]

    fecha_inicio = station_data.iloc[0]['FECHA_INICIO']
    fecha_fin = station_data.iloc[0]["FECHA_FIN"]

    fecha_inicio = pd.to_datetime(fecha_inicio)
    fecha_fin = pd.to_datetime(fecha_fin)

    if (fecha_fin <= pd.Timestamp("2007-12-01") or fecha_inicio >= pd.Timestamp("2024-12-31")) and warnings:
        print(f"Station {station_id} does not have values for the datacube period.")

    station_path = get_aemet_station_path_from_id(station_id)

    df = pd.read_csv(station_path, encoding="latin1", sep=";")
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df = df[df["FECHA"] >= pd.Timestamp("2007-12-01")]
    df = df[df["FECHA"] <= pd.Timestamp("2024-12-31")]
    return df


def process_aemet_station_data(station_data):

    # Convert the FECHA column to datetime format
    station_data["FECHA"] = pd.to_datetime(station_data["FECHA"])
    # Drop columns with all NaN values
    station_data = station_data.dropna(axis=1, how="all") 
    # Drop rows with all Nan values
    station_data = station_data.dropna(axis=0, how="all")
    # If PRECIPITATION column is in station data, transform all "Ip" values to 0.0
    if "PRECIPITACION" in station_data.columns:
        station_data["PRECIPITACION"] = station_data["PRECIPITACION"].replace("Ip", 0.0)
        station_data["PRECIPITACION"] = station_data["PRECIPITACION"].replace("Acum", np.nan)
        station_data["PRECIPITACION"] = station_data["PRECIPITACION"].astype(float)
        station_data["PRECIPITACION"] = station_data["PRECIPITACION"] / 24  # Calculate the daily mean

    # If DIR column is in station data, transform to degrees and replace 99 with NaN
    if "DIR" in station_data.columns:
        station_data["DIR"] = station_data["DIR"].replace(99, np.nan)
        station_data["DIR"] = station_data["DIR"] * 10
        station_data.loc[station_data['DIR'] > 360, 'DIR'] = np.nan

    # Temperatures
    temps_names = ["TMEDIA", "TMIN", "TMAX"]
    for temp_name in temps_names:
        if temp_name in station_data.columns:
            station_data[temp_name] = station_data[temp_name].astype(float) 

    # Pressure
    pres_names = ["PRESMAX", "PRESMIN"]
    for pres_name in pres_names:
        if pres_name in station_data.columns:
            station_data[pres_name] = station_data[pres_name].astype(float) 

    winds = ["VELMEDIA", "RACHA"]
    for wind_name in winds:
        if wind_name in station_data.columns:
            station_data[wind_name] = station_data[wind_name].astype(float)

    return station_data
