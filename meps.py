#%%
import xarray as xr
from siphon.catalog import TDSCatalog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import streamlit as st
import datetime
import matplotlib.dates as mdates
#
from matplotlib.colors import Normalize
import folium
from branca.colormap import linear

@st.cache_data(ttl=60)
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


@st.cache_data()
def load_meps_for_location(file_path=None, altitude_min=0, altitude_max=3000):
    """
    file_path=None
    altitude_min=0
    altitude_max=3000
    """
    if file_path is None:
        file_path = find_latest_meps_file()

    x_range= "[200:1:300]"
    y_range= "[400:1:500]"
    time_range = "[0:1:66]"
    hybrid_range = "[25:1:64]"
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

    subset = xr.open_dataset(path, cache=True)
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
    
    
    altitude = hybrid_to_height(subset).mean("time").squeeze().mean("x").mean("y")
    subset = subset.assign_coords(altitude=('hybrid', altitude.data))
    subset = subset.swap_dims({'hybrid': 'altitude'})

    # filter subset on altitude ranges
    subset = subset.where((subset.altitude >= altitude_min) & (subset.altitude <= altitude_max), drop=True).squeeze()

    wind_speed = np.sqrt(subset['x_wind_ml']**2 + subset['y_wind_ml']**2)
    subset = subset.assign(wind_speed=(('time', 'altitude','y','x'), wind_speed.data))

    thermal_temp_diff = compute_thermal_temp_difference(subset)
    subset = subset.assign(thermal_temp_diff=(('time', 'altitude','y','x'), thermal_temp_diff.data))

    # Find the indices where the thermal temperature difference is zero or negative
    indices = (thermal_temp_diff > 0).argmax(dim="altitude")
    # Get the altitudes corresponding to these indices
    thermal_top = subset.altitude[indices]
    subset = subset.assign(thermal_top=(('time', 'y', 'x'), thermal_top.data))

    subset = subset.set_coords(["latitude", "longitude"])

    return subset


#%%
def compute_thermal_temp_difference(subset):
    lapse_rate = 0.0098
    air_temp = (subset['air_temperature_ml']-273.3).ffill(dim='altitude')
    # Plot airtemp on a map

    ground_temp = 3+ air_temp.where(air_temp.altitude == air_temp.altitude.min())
    ground_temp_filled = ground_temp.bfill(dim='altitude')
    temp_parcel = ground_temp_filled - lapse_rate * air_temp.altitude
    #temp_parcel.plot()
    thermal_temp_diff = (temp_parcel - air_temp).clip(min=0)
    return thermal_temp_diff

@st.cache_data(ttl=60)
def create_wind_map(_subset,  x_target, y_target, altitude_max=3000, date_start=None, date_end=None):
    """
    altitude_max = 3000
    date_start = None
    date_end = None
    """
    subset = _subset
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
    new_altitude = np.arange(subset.altitude.min(), altitude_max, altitude_max/20)
    if date_start is None:
        date_start = subset.time.min().values
    if date_end is None:
        date_end = subset.time.max().values
    new_timestamps = pd.date_range(date_start, date_end, 20)
    #subset
    windplot_data = subset.sel(x=x_target, y=y_target, method="nearest").interp(altitude=new_altitude, time=new_timestamps)

    contourf = ax.contourf(windplot_data.time, windplot_data.altitude, windplot_data.thermal_temp_diff.T, cmap=tempcolors, alpha=0.5, vmin=0, vmax=4)
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

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # normalize wind speed for color mapping
    norm = plt.Normalize(0, 20)

    # add numerical labels to the plot
    for x, t in enumerate(windplot_data.time.values):
        for y, alt in enumerate(windplot_data.altitude.values):
            color = windcolors(norm(windplot_data.wind_speed[x,y]))
            ax.text(t, alt-50, f"{windplot_data.wind_speed[x,y]:.1f}", size=6, color=color)
    plt.title(f"Wind and thermals in Sogndal {date_start.strftime('%Y-%m-%d')}")
    plt.yscale("linear")

    # Return the figure object
    return fig

#%%
@st.cache_data(ttl=7200)
def create_sounding(_subset, date, hour, x_target, y_target, altitude_max=3000):
    """
    date = "2024-05-12"
    hour = "15"
    x_target = 5
    y_target = 5
    """
    subset = _subset
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
    time_str = f"{date} {hour}:00:00"
    # find x and y values cloeset to given latitude and longitude

    ds_time = subset.sel(time=time_str, x=x_target,y=y_target, method="nearest")
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
    ax.set_title(f'Temperature Profile and Dry Adiabatic Lapse Rate for {date} {hour}:00')
    ax.legend(title='Time')
    xmin, xmax = ds_time['air_temperature_ml'].min().values-273.3, ds_time['air_temperature_ml'].max().values-273.3+3
    ax.set_xlim(xmin, xmax)
    ax.grid(True)

    # Return the figure object
    return fig



@st.cache_data(ttl=7200)
def build_map(_subset, date=None, hour=None, x_target=None, y_target=None):
    """
    date = "2024-05-12"
    hour = "15"
    """
    subset = _subset
    # Get the thermal_top data for a specific time
    thermal_top_values = subset.thermal_top.sel(time=f"{date}T{hour}").values

    # Get the latitude and longitude values from the dataset
    latitude_values = subset.latitude.values
    longitude_values = subset.longitude.values

    # Normalize the data to 0-1
    col_thermal_min, col_thermal_max = 0.2, 3000
    norm = Normalize(vmin=col_thermal_min, vmax=col_thermal_max)
    normalized_data = norm(thermal_top_values)

    # Create a color map
    cmap = plt.get_cmap('Reds')
    colored_data = cmap(normalized_data)

    # Create an image from the colored data
    img = np.uint8(colored_data * 255)

    # Get the bounds of the data
    bounds = [[latitude_values.min(), longitude_values.min()], [latitude_values.max(), longitude_values.max()]]

    # Create a map centered at the mean of the latitude and longitude values
    m = folium.Map(location=[latitude_values.mean(), longitude_values.mean()], zoom_start=10)

    # Add the image overlay to the map
    folium.raster_layers.ImageOverlay(img, bounds=bounds, opacity=0.7, mercator_project=True).add_to(m)

    # Create a color map legend
    colormap = linear.YlOrRd_09.scale(col_thermal_min, col_thermal_max)
    colormap.caption = 'Height'
    m.add_child(colormap)

    # Add marker on point of estimates
    lon, lat = subset.sel(x=x_target, y=y_target, method="nearest").latitude.values, subset.sel(x=x_target, y=y_target, method="nearest").longitude.values
    folium.Marker([lon, lat], popup='Forecast Location').add_to(m)
    return m

#%%
import pyproj
def latlon_to_xy(lat, lon):
    crs = pyproj.CRS.from_cf(
        {
            "grid_mapping_name": "lambert_conformal_conic",
            "standard_parallel": [63.3, 63.3],
            "longitude_of_central_meridian": 15.0,
            "latitude_of_projection_origin": 63.3,
            "earth_radius": 6371000.0,
        }
    )
    # Transformer to project from ESPG:4368 (WGS:84) to our lambert_conformal_conic
    proj = pyproj.Proj.from_crs(4326, crs, always_xy=True)
    # Compute projected coordinates of lat/lon point
    X,Y = proj.transform(lon,lat)
    return X,Y
# %%
def show_forecast():

    col1, col2 = st.columns([1, 1])
    with col1:
        latitude = st.number_input("latitude", value=61.22908)
    with col2:
        longitude = st.number_input("longitude", value=7.09674)
    x_target, y_target = latlon_to_xy(latitude, longitude)
    with st.spinner('Fetching data...'):
        if "file_path" not in st.session_state:
            st.session_state.file_path = find_latest_meps_file()
        
        subset = load_meps_for_location(st.session_state.file_path)

    def date_controls():
        
        start_stop_time = [subset.time.min().values.astype('M8[ms]').astype('O'), subset.time.max().values.astype('M8[ms]').astype('O')]
        now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)

        if "forecast_date" not in st.session_state:
            st.session_state.forecast_date = now.date()
        if "forecast_time" not in st.session_state:
            st.session_state.forecast_time = datetime.time(14,0)
        if "forecast_length" not in st.session_state:
            st.session_state.forecast_length = 1
        if "altitude_max" not in st.session_state:
            st.session_state.altitude_max = 3000
        col1, col_date, col_time, col3 = st.columns([0.2,0.6,0.2,0.2])

        with col1:
            if st.button("⏮️", use_container_width=True):
                st.session_state.forecast_date -= datetime.timedelta(days=1)
        with col3:
            if st.button("⏭️", use_container_width=True, disabled=(st.session_state.forecast_date == start_stop_time[1])):
                st.session_state.forecast_date += datetime.timedelta(days=1)
        with col_date:
            st.session_state.forecast_date = st.date_input(
                "Start date", 
                value=st.session_state.forecast_date, 
                min_value=start_stop_time[0], 
                max_value=start_stop_time[1], 
                label_visibility="collapsed",
                disabled=True
                )
        with col_time:
            st.session_state.forecast_time = st.time_input("Start time", value=st.session_state.forecast_time, step=3600,disabled=False,label_visibility="collapsed")

    date_controls()
    time_start = datetime.time(0, 0)
    # convert subset.attrs['min_time']='2024-05-11T06:00:00Z' into datetime
    min_time = datetime.datetime.strptime(subset.attrs['min_time'], "%Y-%m-%dT%H:%M:%SZ")
    date_start = datetime.datetime.combine(st.session_state.forecast_date, time_start)
    date_start = max(date_start, min_time)
    date_end= datetime.datetime.combine(st.session_state.forecast_date+datetime.timedelta(days=st.session_state.forecast_length), datetime.time(0, 0))

    ## MAP
    with st.expander("Map", expanded=True):
        m = build_map(subset, date=st.session_state.forecast_date, hour=st.session_state.forecast_time, x_target=x_target, y_target=y_target)
        from streamlit_folium import folium_static
        folium_static(m)

    wind_fig = create_wind_map(
                subset,
                date_start=date_start, 
                date_end=date_end, 
                altitude_max=st.session_state.altitude_max,
                x_target=x_target,
                y_target=y_target)
    st.pyplot(wind_fig)
    

    with st.expander("More settings", expanded=False):
        st.session_state.forecast_length = st.number_input("multiday", 1, 3, 1, step=1,)
        st.session_state.altitude_max = st.number_input("Max altitude", 0, 4000, 3000, step=500)
    
    ############################
    ######### SOUNDING #########
    ############################
    st.markdown("---")
    with st.expander("Sounding", expanded=False):
        date = datetime.datetime.combine(st.session_state.forecast_date, st.session_state.forecast_time)

        with st.spinner('Building sounding...'):
            sounding_fig = create_sounding(
                subset, 
                date=date.date(), 
                hour=date.hour, 
                altitude_max=st.session_state.altitude_max,
                x_target=x_target,
                y_target=y_target)
        st.pyplot(sounding_fig)

    st.markdown("Wind and sounding data from MEPS model (main model used by met.no). Thermal (green) is assuming ground temperature is 3 degrees higher than surrounding air. The location for both wind and sounding plot is Sogndal (61.22, 7.09). Ive probably made many errors in this process.")

    # Download new forecast if available
    st.session_state.file_path = find_latest_meps_file()
    subset = load_meps_for_location(st.session_state.file_path)


if __name__ == "__main__":
    lat = 61.22908
    lon = 7.09674
    x_target, y_target = latlon_to_xy(lat, lon)
    
    dataset_file_path = find_latest_meps_file()
    local=True
    if local:
        subset = xr.open_dataset("subset.nc")
    else:
        subset = load_meps_for_location()
        subset.to_netcdf("subset.nc")

    build_map(subset, date="2024-05-13", hour="15")

    wind_fig = create_wind_map(subset, altitude_max=3000,x_target=x_target, y_target=y_target)

    # Plot thermal top on a map for a specific time
    #subset.sel(time=subset.time.min()).thermal_top.plot()
    sounding_fig = create_sounding(subset, date="2024-05-12", hour=15, x_target=x_target, y_target=y_target)

    

