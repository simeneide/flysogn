#%%
import xarray as xr
from siphon.catalog import TDSCatalog
import numpy as np
import streamlit as st
import datetime
from datetime import datetime
from siphon.catalog import TDSCatalog


# %%
lat = 61.22908
lon = 7.09674
tol=0.3
@st.cache_data(ttl=7200)
@st.cache_data(ttl=60)
def find_latest_meps_file():
    today = datetime.today()
    catalog_url = f"https://thredds.met.no/thredds/catalog/meps25epsarchive/{today.year}/{today.month:02d}/{today.day:02d}/catalog.xml"
    file_url_base = f"https://thredds.met.no/thredds/dodsC/meps25epsarchive/{today.year}/{today.month:02d}/{today.day:02d}"
    # Get the datasets from the catalog
    catalog = TDSCatalog(catalog_url)
    datasets = [s for s in catalog.datasets if "meps_det_ml" in s]
    dataset_file_path = f"{file_url_base}/{sorted(datasets)[-1]}"
    return dataset_file_path


#%%
def load_meps_for_location(dataset_file_path=None, altitude_min=0, altitude_max=3000):
    """
    dataset_file_path=None
    altitude_min=0
    altitude_max=3000
    """
    if dataset_file_path is None:
        dataset_file_path = find_latest_meps_file()

    dataset = xr.open_dataset(dataset_file_path, cache=True)
    """
    # Dataset has the following format:
    <xarray.Dataset> Size: 212GB
    Dimensions:                                           (height: 1, hybrid: 65,
                                                        x: 949, y: 1069, time: 67)
    Coordinates:
    * height                                            (height) float32 4B 0.0
    * hybrid                                            (hybrid) float64 520B 0...
    * x                                                 (x) float32 4kB -1.06e+...
    * y                                                 (y) float32 4kB -1.333e...
    * time                                              (time) datetime64[ns] 536B ...
        longitude                                         (y, x) float64 8MB 0.27...
        latitude                                          (y, x) float64 8MB 50.3...
    Data variables: (12/18)
        forecast_reference_time                           datetime64[ns] 8B ...
        p0                                                float64 8B ...
        ap                                                (hybrid) float64 520B ...
        b                                                 (hybrid) float64 520B ...
        projection_lambert                                int32 4B ...
        specific_humidity_ml                              (time, hybrid, y, x) float32 18GB ...
        ...                                                ...
        cloud_area_fraction_ml                            (time, hybrid, y, x) float32 18GB ...
        air_temperature_ml                                (time, hybrid, y, x) float32 18GB ...
        x_wind_ml                                         (time, hybrid, y, x) float32 18GB ...
        y_wind_ml                                         (time, hybrid, y, x) float32 18GB ...
        upward_air_velocity_ml                            (time, hybrid, y, x) float32 18GB ...
        surface_air_pressure                              (time, height, y, x) float32 272MB ...
    Attributes: (12/40)
        min_time:                    2024-05-05T00:00:00Z
        geospatial_lat_min:          49.8
        geospatial_lat_max:          75.2
        geospatial_lon_min:          -18.1
        geospatial_lon_max:          54.2
        comment:                     For more information, please visit https://g...
        ...                          ...
        publisher_url:               https://data.met.no
        publisher_name:              Norwegian Meteorological Institute
        summary:                     This file contains model level parameters fr...
        summary_no:                  Denne filen inneholder modelnivåparametere f...
        title:                       Meps 2.5Km deterministic model level paramet...
        related_dataset:             no.met:8c94c7de-6328-4113-9e77-8f090999fab9 ...
    """

    # subset sogndal area:
    subset = dataset.sel(
        x=slice(dataset.x.quantile(0.265), dataset.x.quantile(0.28)), 
        y=slice(dataset.y.quantile(0.42), dataset.y.quantile(0.43))
    )
    subset = subset.drop_vars(['projection_lambert', 'forecast_reference_time', 'mass_fraction_of_cloud_condensed_water_in_air_ml','mass_fraction_of_cloud_ice_in_air_ml','mass_fraction_of_snow_in_air_ml','mass_fraction_of_rain_in_air_ml','mass_fraction_of_graupel_in_air_ml','turbulent_kinetic_energy_ml',])

    #subset.load()

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
    subset = subset.where((subset.altitude >= altitude_min) & (subset.altitude <= altitude_max), drop=True).squeeze()

    wind_speed = np.sqrt(subset['x_wind_ml']**2 + subset['y_wind_ml']**2)
    subset = subset.assign_coords(wind_speed=(('time', 'altitude','y','x'), wind_speed.data))

    return subset

subset = load_meps_for_location()

def compute_thermal_temp_difference(subset):
    lapse_rate = 0.0098
    air_temp = (subset['air_temperature_ml']-273.3) #.ffill(dim='altitude')
    ground_temp = 3+ air_temp.where(air_temp.altitude == air_temp.altitude.min())
    ground_temp_filled = ground_temp.bfill(dim='altitude')
    temp_parcel = ground_temp_filled - lapse_rate * air_temp.altitude
    #temp_parcel.plot()
    thermal_temp_diff = (temp_parcel - air_temp).clip(min=0)
    return thermal_temp_diff

thermal_diff_df = compute_thermal_temp_difference(subset)
#%% plot latitude and longitude points on  a leaflet map
import folium

# Get the latitude and longitude values from the dataset
latitude_values = subset.latitude.values.flatten()
longitude_values = subset.longitude.values.flatten()

# Create a map centered at the mean of the latitude and longitude values
from folium.plugins import HeatMap

# Create a map centered at the mean of the latitude and longitude values
m = folium.Map(location=[latitude_values.mean(), longitude_values.mean()], zoom_start=4)

# Prepare data for the heatmap
data = [[lat, lon] for lat, lon in zip(latitude_values, longitude_values)]

# Add a heatmap to the map with reduced opacity
HeatMap(data, opacity=0.1).add_to(m)  # Adjust the opacity as needed

# Display the map
m
# %%
