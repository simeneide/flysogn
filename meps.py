#%%
import xarray as xr
from siphon.catalog import TDSCatalog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

def load_meps_for_location(lat, lon, tol=0.1, altitude_min=0, altitude_max=3000):

    # The MEPS dataset: https://github.com/metno/NWPdocs/wiki/MEPS-dataset
    catalog_url = "https://thredds.met.no/thredds/catalog/mepslatest/catalog.xml"
    # Create a catalog object
    catalog = TDSCatalog(catalog_url)
    datasets = [s for s in catalog.datasets if "meps_det_vc_2_5km" in s]
    latest_dataset = sorted(datasets)[-1]

    ds_path = f"https://thredds.met.no/thredds/dodsC/mepslatest/{latest_dataset}"
    dataset = xr.open_dataset(ds_path) # , engine="netcdf4"

    # Filter dataset on lat and lon
    subset = dataset.where(
        (np.abs(dataset.latitude - lat) < tol) & (np.abs(dataset.longitude - lon) < tol), 
        drop=True
    )
    subset.load()

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

    altitude = hybrid_to_height(subset).mean("time").squeeze()#.mean("ensemble_member")
    subset = subset.assign_coords(altitude=('hybrid', altitude.data))
    subset = subset.swap_dims({'hybrid': 'altitude'})

    # filter subset on altitude ranges
    subset = subset.where((subset.altitude >= altitude_min) & (subset.altitude <= altitude_max), drop=True).squeeze()

    wind_speed = np.sqrt(subset['x_wind_ml']**2 + subset['y_wind_ml']**2)
    subset = subset.assign_coords(wind_speed=(('time', 'altitude'), wind_speed.data))
    return subset

#%%
def compute_thermal_temp_difference(subset):
    lapse_rate = 0.0098
    air_temp = (subset['air_temperature_ml']-273.3).ffill(dim='altitude')
    ground_temp = 3+ air_temp.where(air_temp.altitude == air_temp.altitude.min())
    ground_temp_filled = ground_temp.bfill(dim='altitude')
    temp_parcel = ground_temp_filled - lapse_rate * air_temp.altitude
    #temp_parcel.plot()
    thermal_temp_diff = (temp_parcel - air_temp).clip(min=0)
    return thermal_temp_diff


def create_wind_map(subset, altitude_max=3000, date_start=None, date_end=None):
    """
    altitude_max = 3000
    date_start = None
    date_end = None
    """
    windcolors = mcolors.LinearSegmentedColormap.from_list("", ["grey", "green","darkgreen","yellow","orange","darkorange","red", "darkred", "violet","darkviolet"],)

    # build colorscale for thermal temperature difference
    colors = [(1, 1, 1),  # white
              (1, 1, 1),  # white
            (1, 0, 0),  # red
            (0.58, 0, 0.83)]  # violet
    positions = [0, 0.05,0.5, 1]  # transition points

    # Create the colormap
    tempcolors = mcolors.LinearSegmentedColormap.from_list("", list(zip(positions, colors)))

    # Create a figure object
    fig, ax = plt.subplots(figsize=(15, 7))
    new_altitude = np.arange(0, altitude_max, altitude_max/20)
    if date_start is None:
        date_start = subset.time.min().values
    if date_end is None:
        date_end = subset.time.max().values
    new_timestamps = pd.date_range(date_start, date_end, 20)
    windplot_data = subset.interp(altitude=new_altitude, time=new_timestamps)

    # Build thermal temperature difference plot
    thermal_temp_diff = compute_thermal_temp_difference(subset)
    thermal_temp_diff = thermal_temp_diff.interp(altitude=new_altitude, time=new_timestamps).bfill(dim='altitude')
    contourf = ax.contourf(windplot_data.time, windplot_data.altitude, thermal_temp_diff.T, cmap=tempcolors, alpha=0.5, vmin=0, vmax=4)
    fig.colorbar(contourf, ax=ax, label='Thermal Temperature Difference (°C)', pad=0.01, orientation='vertical')
    
    # Wind quiver plot
    quiverplot = windplot_data.plot.quiver(
        x='time', y='altitude', u='x_wind_ml', v='y_wind_ml', 
        hue="wind_speed", 
        cmap = windcolors,
        vmin=2, vmax=20,
        pivot="middle",# headwidth=4, headlength=6,
        ax=ax  # Add this line to plot on the created axes 
    )
    quiverplot.colorbar.set_label("Wind Speed  [m/s]")
    quiverplot.colorbar.pad = 0.01

    # normalize wind speed for color mapping
    norm = plt.Normalize(0, 20)

    # add numerical labels to the plot
    for x, t in enumerate(windplot_data.time.values):
        for y, alt in enumerate(windplot_data.altitude.values):
            color = windcolors(norm(windplot_data.wind_speed[x,y]))
            ax.text(t, alt-50, f"{windplot_data.wind_speed[x,y]:.1f}", size=6, color=color)
    plt.title("Wind and thermals in Sogndal")
    plt.yscale("linear")

    # Return the figure object
    return fig

#%%
def create_sounding(subset, date, hour, hour_end=None, altitude_max=3000):
    lapse_rate = 0.0098 # in degrees Celsius per meter
    subset = subset.where(subset.altitude< altitude_max,drop=True)
    # Create a figure object
    fig, ax = plt.subplots()

    # Define the dry adiabatic lapse rate
    def add_dry_adiabatic_lines(ds):
        # Define a range of temperatures at sea level
        T0 = np.arange(-40, 40, 5)  # temperatures from -40°C to 40°C in steps of 10°C

        # Create a 2D grid of temperatures and altitudes
        T0, altitude = np.meshgrid(T0, ds.altitude)

        # Calculate the temperatures at each altitude
        T_adiabatic = T0 - lapse_rate * altitude

        # Plot the dry adiabatic lines
        for i in range(T0.shape[1]):
            ax.plot(T_adiabatic[:, i], ds.altitude, 'r:', alpha=0.5)

    # Plot the actual temperature profiles
    if hour_end is None:
        hour_end = hour+1
    for h in range(hour, hour_end):  # 10 to 17 inclusive
        time_str = f"{date} {h}:00:00"
        ds_time = subset.sel(time=time_str, method="nearest")
        T = (ds_time['air_temperature_ml'].values-273.3)  # in degrees Celsius
        ax.plot(T, ds_time.altitude, label=f"temp {pd.to_datetime(time_str).strftime('%H:%M')}")

        # Define the surface temperature
        T_surface = T[-1]+3
        T_parcel = T_surface - lapse_rate * ds_time.altitude

        # Plot the temperature of the rising air parcel
        filter = T_parcel>T
        ax.plot(T_parcel[filter], ds_time.altitude[filter], 'g-', label='Rising air parcel',color="green")

    add_dry_adiabatic_lines(ds_time)

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Temperature Profile and Dry Adiabatic Lapse Rate for {date}')
    ax.legend(title='Time')
    xmin, xmax = ds_time['air_temperature_ml'].min().values-273.3, ds_time['air_temperature_ml'].max().values-273.3+3
    ax.set_xlim(xmin, xmax)
    ax.grid(True)

    # Return the figure object
    return fig
# %%

if __name__ == "__main__":
    lat = 61.22908
    lon = 7.09674
    subset = load_meps_for_location(lat, lon, tol=0.1, altitude_min=0, altitude_max=3000)
    wind_fig = create_wind_map(subset, altitude_max=3000)
    sounding_fig = create_sounding(subset, date="2024-04-02", hour=15)