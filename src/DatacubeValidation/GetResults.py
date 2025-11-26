import pickle
import os
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.config import AEMET_VALIDATION_RESUTLS_DIR
from src.DatacubeValidation.stations import get_valid_aemet_stations_geodataframe

def load_result_geodataframe(metric = "MAE"):
    """
    Load the results of the validation process into a GeoDataFrame.
    Args:
        metric (str): The metric to load. Options are "MAE" or "NormalizedMAE".
        
    Returns:
        gdf (GeoDataFrame): A GeoDataFrame containing the validation results.
    """

    gdf = get_valid_aemet_stations_geodataframe()
    features_to_validate = ["TMAX", "TMIN", "TMEDIA", "PRECIPITACION", "DIR", "VELMEDIA", "RACHA", "PRESMAX", "PRESMIN"]
    for feature in features_to_validate:
        gdf[feature] = None

    minmax_values = pickle.load(open(os.path.join(AEMET_VALIDATION_RESUTLS_DIR, f"AEMET_minmax_dict.pkl"), "rb"))
    for i, row in gdf.iterrows():
        id = row["INDICATIVO"]
        MAE_results = pickle.load(open(os.path.join(AEMET_VALIDATION_RESUTLS_DIR, "stations_MAE", f"validation_MAE_{row['INDICATIVO']}.pkl"), "rb"))
        for feature in features_to_validate:
            if feature in MAE_results:
                if metric == "MAE":
                    gdf.at[i, feature] = MAE_results[feature]

                elif metric == "NormalizedMAE":
                    gdf.at[i, feature] = MAE_results[feature] / (minmax_values[id][f"{feature}_max"] - minmax_values[id][f"{feature}_min"])

    for feature in features_to_validate:
        gdf[feature] = gdf[feature].astype(float)

    return gdf




def plot_results(gdf, feature_name, ax, title, error_metric = "MAE"):

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.1)  
    cax.tick_params(labelsize=20)
    gdf_plotting = gdf.to_crs(epsg=3857)  # Change crs for plotting

    gdf_plotting.plot(
        column=feature_name, 
        cmap="plasma_r",  
        legend=True, 
        ax=ax, 
        legend_kwds={'label': "", 'orientation': "horizontal"},
        alpha=0.8, 
        cax=cax,
        markersize=70,
        vmin = None if error_metric == "MAE" else 0,
        vmax = None if error_metric == "MAE" else .2,
    )
        
    ctx.add_basemap(ax, crs=gdf_plotting.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels)

    ax.set_title(title, fontsize=30)

    ax.set_axis_off()  
