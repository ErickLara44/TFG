import xarray as xr
import os
import argparse
import time

def export_static_datacube(input_path: str, output_path: str):
    """
    Reads a large Datacube (30GB) and extracts ONLY the static variables
    (those without a 'time' dimension), creating a much smaller footprint
    Datacube (~100MB) suitable for production deployment in Hugging Face or Vercel.
    """
    print(f"Loading heavy dataset from: {input_path}")
    start_time = time.time()
    
    # We open it, but don't load everything into memory (Dask handles it lazy)
    ds = xr.open_dataset(input_path)
    
    static_vars = []
    
    for var_name, var_data in ds.data_vars.items():
        if 'time' not in var_data.dims:
            static_vars.append(var_name)
    
    print(f"\\nFound {len(static_vars)} static variables:")
    for v in static_vars:
        print(f" - {v}")
    
    print(f"\\nExtracting static dataset...")
    # Create a new dataset containing only these variables
    ds_static = ds[static_vars]
    
    # If the user wants to test, they might also need the spatial coordinates (x, y) 
    # to remain exactly as they are. Xarray handles this automatically.
    
    print(f"Saving lightweight dataset to: {output_path}")
    # Save the new static datacube to disk
    ds_static.to_netcdf(output_path)
    
    end_time = time.time()
    
    # Check original vs new size
    orig_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\\n✅ Done in {end_time - start_time:.2f} seconds!")
    print(f"Original size: {orig_size_mb:.2f} MB")
    print(f"New static size: {new_size_mb:.2f} MB")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts static variables from a time-series Datacube")
    parser.add_argument("--input", type=str, default="/Users/erickmollinedolara/Erick/Uni/TFG/Datacube/datacube_spain_2019_2024.nc",
                        help="Path to the original large NetCDF Datacube")
    parser.add_argument("--output", type=str, default="data/datacube_static.nc",
                        help="Path where the new lightweight Datacube will be saved")
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    export_static_datacube(args.input, args.output)
