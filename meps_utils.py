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

def compute_temperature_above_surface(subset):
    mask = subset['hybrid_altitude'] <= subset['elevation']
    first_above_surface_idx = mask.argmax(dim='altitude', skipna=True)
    valid_mask = mask.any(dim='altitude')
    first_above_surface_idx = xr.where(valid_mask, first_above_surface_idx, len(subset.altitude)-1)
    # Convert the index to integer position because take_along_axis expects integer indices
    first_above_surface_idx = first_above_surface_idx.astype(int)

    # Step 4: Extract the corresponding air temperatures using advanced indexing
    # We need to align dimensions for take_along_axis, ensuring 'altitude' is at the correct position
    air_temperature_ml = subset['air_temperature_ml'].transpose('altitude','time', 'y', 'x')

    temperature_above_surface = np.take_along_axis(air_temperature_ml.values, first_above_surface_idx.values[np.newaxis,:, :, :], axis=0).squeeze()
    temperature_above_surface = xr.DataArray(temperature_above_surface, dims=('time', 'y', 'x'))
    return temperature_above_surface

def compute_thermal_temp_difference(subset):
    lapse_rate = 0.0098

    air_temp = (subset['air_temperature_ml']-273.3)

    subset['temperature_above_surface'] = compute_temperature_above_surface(subset)-273.3
    use_internal_temp=True
    if use_internal_temp:
        ground_temp = subset['temperature_above_surface']+3
    else:
        ground_temp = subset.air_temperature_0m-273.3

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
        "specific_humidity_ml": f"{time_range}{hybrid_range}{y_range}{x_range}",
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
        "air_temperature_2m": f"{time_range}[0:1:0]{y_range}{x_range}",
    } 
    file_path_surf = f"{file_path.replace('meps_det_ml','meps_det_sfc')}?{','.join(f'{k}{v}' for k, v in surf_params.items())}"
    #file_path_surf = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/2024/06/16/meps_det_sfc_20240616T09Z.ncml'
    # Load surface parameters and merge into the main dataset
    surf = xr.open_dataset(file_path_surf, cache=True)
    # Convert the surface geopotential to elevation
    elevation = (surf.surface_geopotential / 9.80665).squeeze()
    #elevation.plot()
    subset['elevation'] = elevation
    air_temperature_0m = surf.air_temperature_0m.squeeze()
    subset['air_temperature_0m'] = surf.air_temperature_0m.squeeze()
    subset['air_temperature_2m'] = surf.air_temperature_2m.squeeze()
    # (surf.air_temperature_2m-surf.air_temperature_0m).isel(time=14).plot()

    # Get difference between air temperatures and plot on map

    # subset.elevation.plot()
    # subset.air_temperature_0m.plot()
    #%%
    def new_hybrid_to_height_MEPSCODE(subset):
        nc = subset
        nl = nc['hybrid'].size
        ny = nc['y'     ].size
        nx = nc['x'     ].size

        ap   = nc.variables['ap'][:]
        b    = nc.variables['b' ][:]
        ps   = nc.variables['surface_air_pressure'][0,0,:,:]
        tair = nc.variables['air_temperature_ml'  ][0,:,:,:]
        qair = nc.variables['specific_humidity_ml'][0,:,:,:]

        t_virt = tair*(1 + 0.61*qair)

        ap_half = [0.0]
        b__half = [1.0]
        for (ak,bk) in zip(ap[::-1], b[::-1]):
            ap_half.append(2*ak - ap_half[-1])
            b__half.append(2*bk - b__half[-1])

        ap_half = np.array(ap_half)[::-1]
        b__half = np.array(b__half)[::-1]

        # Formula to calculate pressure from hybrid: p(n,k,j,i) = ap(k) + b(k)*ps(n,j,i)"
        # Note that k = 0 is top of atmosphere (ToA), and k = 64 is the lowest model level
        pressure_at_k_half = np.empty((nl+1,ny,nx), ps.dtype)
        for l,ak,bk in zip(range(ap_half.size), ap_half, b__half):
            pressure_at_k_half[l,:,:] = ak + (bk*ps)

        R = 287.058
        g = 9.81

        # Compute half-level heights
        height_at_k_half = np.empty_like(pressure_at_k_half)
        height_at_k_half[-1,:,:] = 0
        for l in range(nl - 1, 0, -1):
            height_at_k_half[l,:,:] = height_at_k_half[l+1,:,:] + (R*t_virt[l,:,:]/g)*np.log(pressure_at_k_half[l+1,:,:]/pressure_at_k_half[l,:,:])

        # Compute full-level heights
        height_at_k      = np.empty_like(t_virt)
        for l in range(nl - 1, 0, -1):
            height_at_k[l,:,:] = 0.5*(height_at_k_half[l+1,:,:] + height_at_k_half[l,:,:])

        height_at_k[0,:,:] = height_at_k_half[1,:,:] + (R*t_virt[0,:,:]/g)*np.log(2)

    def hybrid_to_height_timevec(subset):
        nc = subset
        # Adjusted to include the full time dimension
        nt = nc['time'].size
        nl = nc['hybrid'].size
        ny = nc['y'     ].size
        nx = nc['x'     ].size

        ap   = nc.variables['ap'][:]
        b    = nc.variables['b' ][:]
        ps = nc.variables['surface_air_pressure'][:, 0, :, :]
        tair = nc.variables['air_temperature_ml'][:, :, :, :]
        qair = nc.variables['specific_humidity_ml'][:, :, :, :]

        t_virt = tair * (1 + 0.61 * qair)

        # No changes needed here as these are independent of the time dimension
        ap_half = [0.0]
        b_half = [1.0]
        for (ak, bk) in zip(ap[::-1], b[::-1]):
            ap_half.append(2 * ak - ap_half[-1])
            b_half.append(2 * bk - b_half[-1])

        ap_half = np.array(ap_half)[::-1]
        b_half = np.array(b_half)[::-1]

        # Adjusted to include the time dimension in pressure calculation
        nt = ps.shape[0]  # Number of time steps
        pressure_at_k_half = np.empty((nt, nl + 1, ny, nx), dtype=ps.dtype)
        for l, ak, bk in zip(range(ap_half.size), ap_half, b_half):
            pressure_at_k_half[:, l, :, :] = ak + (bk * ps)

        R = 287.058
        g = 9.81

        # Vectorized computation of half-level heights
        height_at_k_half = np.empty_like(pressure_at_k_half)
        height_at_k_half[:, -1, :, :] = 0  # Initialize the lowest level to 0 for all time steps
        for l in range(nl - 1, 0, -1):
            height_at_k_half[:, l, :, :] = height_at_k_half[:, l + 1, :, :] + (R * t_virt[:, l, :, :] / g) * np.log(pressure_at_k_half[:, l + 1, :, :] / pressure_at_k_half[:, l, :, :])

        # Vectorized computation of full-level heights
        height_at_k = np.empty_like(t_virt)
        for l in range(nl - 1, 0, -1):
            height_at_k[:, l, :, :] = 0.5 * (height_at_k_half[:, l + 1, :, :] + height_at_k_half[:, l, :, :])

        height_at_k[:, 0, :, :] = height_at_k_half[:, 1, :, :] + (R * t_virt[:, 0, :, :] / g) * np.log(2)
        return height_at_k
    
    dims = ('time', 'hybrid', 'y', 'x')
    altitude = hybrid_to_height_timevec(subset)
    subset['hybrid_altitude'] = (dims, altitude)
    altitude_coord = subset['hybrid_altitude'].mean("time").squeeze().mean("x").mean("y")
    subset = subset.assign_coords(altitude=('hybrid', altitude_coord.data))
    subset = subset.swap_dims({'hybrid': 'altitude'})

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

    # Compute thermal top above the surface
    thermal_top_above_surface = (thermal_top - subset.elevation).clip(min=0)
    subset = subset.assign(thermal_top_above_surface=(('time', 'y', 'x'), thermal_top_above_surface.data))
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