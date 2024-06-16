import xarray as xr
from siphon.catalog import TDSCatalog
import numpy as np
import datetime
from google.cloud import storage
import os

#%%
def find_latest_meps_file():
    # The MEPS dataset: https://github.com/metno/NWPdocs/wiki/MEPS-dataset
    today = datetime.datetime.today()
    catalog_url = f"https://thredds.met.no/thredds/catalog/meps25epsarchive/{today.year}/{today.month:02d}/{today.day:02d}/catalog.xml"
    file_url_base = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{today.year}/{today.month:02d}/{today.day:02d}"
    # Get the datasets from the catalog
    catalog = TDSCatalog(catalog_url)
    datasets = [s for s in catalog.datasets if "meps_det_ml" in s]
    file_path = f"{file_url_base}/{sorted(datasets)[-1]}"
    return file_path

def compute_thermal_temp_difference(subset):
    lapse_rate = 0.0098
    ground_temp = subset.air_temperature_0m-273.3
    air_temp = (subset['air_temperature_ml']-273.3)#.ffill(dim='altitude')

    # dimensions
    # 'air_temperature_ml'  altitude: 4 y: 3, x: 3
    # 'elevation'                       y: 3  x: 3
    # 'altitude'            altitude: 4

    # broadcast ground temperature to all altitudes, but let it decrease by lapse rate
    altitude_diff = subset.altitude - subset.elevation
    altitude_diff = altitude_diff.where(altitude_diff >= 0, 0)
    temp_decrease = lapse_rate * altitude_diff
    ground_parcel_temp = ground_temp - temp_decrease
    thermal_temp_diff = (ground_parcel_temp - air_temp).clip(min=0)
    return thermal_temp_diff

def process_meps_file(file_path=None):
    if file_path is None:
        file_path = find_latest_meps_file()

    x_range= "[220:1:300]"
    y_range= "[420:1:500]"
    time_range = "[0:1:66]"
    hybrid_range = "[20:1:64]"
    height_range = "[0:1:0]"

    params = {
        "x": x_range,
        "y": y_range,
        "time": time_range,
        "hybrid": hybrid_range,
        "height": height_range,
        "longitude": f"{y_range}{x_range}",
        "latitude": f"{y_range}{x_range}",
        "air_temperature_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
        "ap" : f"{hybrid_range}",
        "b" : f"{hybrid_range}",
        "surface_air_pressure": f"{time_range}{height_range}{y_range}{x_range}",
        "x_wind_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
        "y_wind_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
    }

    path = f"{file_path}?{','.join(f'{k}{v}' for k, v in params.items())}"

    subset = xr.open_dataset(path) # , cache=True
    #subset.load()

    #%% get geopotential
    time_range_sfc = "[0:1:0]"
    surf_params = {
        "x": x_range,
        "y": y_range,
        "time": f"{time_range}",
        "surface_geopotential": f"{time_range_sfc}[0:1:0]{y_range}{x_range}",
        "air_temperature_0m": f"{time_range}[0:1:0]{y_range}{x_range}",
    } 
    file_path_surf = f"{file_path.replace('meps_det_ml','meps_det_sfc')}?{','.join(f'{k}{v}' for k, v in surf_params.items())}"

    # Load surface parameters and merge into the main dataset
    surf = xr.open_dataset(file_path_surf, cache=True)
    # Convert the surface geopotential to elevation
    elevation = (surf.surface_geopotential / 9.80665).squeeze()
    #elevation.plot()
    subset['elevation'] = elevation
    air_temperature_0m = surf.air_temperature_0m.squeeze()
    subset['air_temperature_0m'] = air_temperature_0m
    # subset.elevation.plot()
    #%%
    def hybrid_to_height(ds):
        """
        ds = subset
        """
        # Constants
        R = 287.05  # Gas constant for dry air
        g = 9.80665  # Gravitational acceleration

        # Calculate the pressure at each level
        p = ds['ap'] + ds['b'] * ds['surface_air_pressure']#.mean("ensemble_member")

        # Get the temperature at each level
        T = ds['air_temperature_ml']#.mean("ensemble_member")

        # Calculate the height difference between each level and the surface
        dp = ds['surface_air_pressure'] - p  # Pressure difference
        dT = T - T.isel(hybrid=-1)  # Temperature difference relative to the surface
        dT_mean = 0.5 * (T + T.isel(hybrid=-1))  # Mean temperature

        # Calculate the height using the hypsometric equation
        dz = (R * dT_mean / g) * np.log(ds['surface_air_pressure'] / p)

        return dz
    
    
    altitude = hybrid_to_height(subset).mean("time").squeeze().mean("x").mean("y")
    subset = subset.assign_coords(altitude=('hybrid', altitude.data))
    subset = subset.swap_dims({'hybrid': 'altitude'})

    # filter subset on altitude ranges
    subset = subset.squeeze()

    wind_speed = np.sqrt(subset['x_wind_ml']**2 + subset['y_wind_ml']**2)
    subset = subset.assign(wind_speed=(('time', 'altitude','y','x'), wind_speed.data))

    
    subset['thermal_temp_diff'] = compute_thermal_temp_difference(subset)
    #subset = subset.assign(thermal_temp_diff=(('time', 'altitude','y','x'), thermal_temp_diff.data))

    # Find the indices where the thermal temperature difference is zero or negative
    # Create tiny value at ground level to avoid finding the ground as the thermal top
    thermal_temp_diff = subset['thermal_temp_diff'] 
    thermal_temp_diff = thermal_temp_diff.where(
        (thermal_temp_diff.sum("altitude")>0)|(subset['altitude']!=subset.altitude.min()), 
        thermal_temp_diff + 1e-6)
    indices = (thermal_temp_diff > 0).argmax(dim="altitude")
    # Get the altitudes corresponding to these indices
    thermal_top = subset.altitude[indices]
    subset = subset.assign(thermal_top=(('time', 'y', 'x'), thermal_top.data))
    subset = subset.set_coords(["latitude", "longitude"])

    return subset


### GCP Functions
def create_dir_if_not_exists(file_path):
    # Get the directory name
    dir_name = os.path.dirname(file_path)
    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

class GCPStorage:
    def __init__(self):
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket("vestavind")

    def preprocess_meps_and_upload_to_gcp(self):
        latest_meps_file = find_latest_meps_file()
        timestamp = latest_meps_file.split("meps_det_ml_")[1].replace(".ncml","")
        subset = process_meps_file(latest_meps_file)

        # save subset to disk
        local_file_path = f"model_files/meps_{timestamp}.nc"
        create_dir_if_not_exists(local_file_path)
        subset.to_netcdf(local_file_path)


        # Create a new blob and upload the file's content to GCS
        blob = self.bucket.blob(f'model_files/{timestamp}.nc')
        blob.upload_from_filename(local_file_path)
        # upload to gcp

    def download_latest_model_file(self):

        # List all the blobs in the bucket that start with the prefix
        blobs = self.bucket.list_blobs(prefix="model_files/")

        # Sort the blobs by their updated time (latest first)
        blobs = sorted(blobs, key=lambda blob: blob.updated, reverse=True)

        # Get the latest blob
        latest_blob = blobs[0]

        # Get the local file path
        local_file_path = f"model_files/{latest_blob.name.split('/')[-1]}"

        create_dir_if_not_exists(local_file_path)

        # Check if the file already exists locally
        if os.path.exists(local_file_path):
            print(f"File {local_file_path} already exists locally.")
        else:
            # Download the file's content from GCS
            latest_blob.download_to_filename(local_file_path)
            print(f"File {latest_blob.name} downloaded locally to {local_file_path}.")
        return local_file_path
    
if __name__ == "__main__":
    gcp = GCPStorage()
    gcp.preprocess_meps_and_upload_to_gcp()
    #local_file_path = gcp.download_latest_model_file()
    #subset = xr.open_dataset(local_file_path)
    #subset