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

@st.cache_data(ttl=60)
def find_latest_meps_file():
    # The MEPS dataset: https://github.com/metno/NWPdocs/wiki/MEPS-dataset
    catalog_url = "https://thredds.met.no/thredds/catalog/mepslatest/catalog.xml"
    # Create a catalog object
    catalog = TDSCatalog(catalog_url)
    
    # ensemble_dataset: meps_lagged_6_h_vc_2_5km
    datasets = [s for s in catalog.datasets if "meps_det_vc_2_5km" in s]
    latest_dataset = sorted(datasets)[-1]
    return latest_dataset


@st.cache_data(ttl=7200)
def load_meps_for_location(dataset_file_path, lat, lon, tol=0.1, altitude_min=0, altitude_max=3000):
    """
    lat = 61.22908
    lon = 7.09674
    tol = 0.1

    """
    ds_path = f"https://thredds.met.no/thredds/dodsC/mepslatest/{dataset_file_path}"
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

@st.cache_data(ttl=60)
def create_wind_map(_subset, altitude_max=3000, date_start=None, date_end=None):
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

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # normalize wind speed for color mapping
    norm = plt.Normalize(0, 20)

    # add numerical labels to the plot
    for x, t in enumerate(windplot_data.time.values):
        for y, alt in enumerate(windplot_data.altitude.values):
            color = windcolors(norm(windplot_data.wind_speed[x,y]))
            ax.text(t, alt-50, f"{windplot_data.wind_speed[x,y]:.1f}", size=6, color=color)
    plt.title(f"Wind and thermals in Sogndal {date_start.date()}")
    plt.yscale("linear")

    # Return the figure object
    return fig

#%%
@st.cache_data(ttl=7200)
def create_sounding(_subset, date, hour, hour_end=None, altitude_max=3000):
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
def show_forecast():
    lat = 61.22908
    lon = 7.09674
    
    with st.spinner('Fetching data...'):
        dataset_file_path = find_latest_meps_file()
        subset = load_meps_for_location(dataset_file_path, lat, lon, tol=0.1, altitude_min=0, altitude_max=4000)

    start_stop_time = [subset.time.min().values.astype('M8[ms]').astype('O'), subset.time.max().values.astype('M8[ms]').astype('O')]
    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)

    if "forecast_date" not in st.session_state:
        st.session_state.forecast_date = now.date()
    if "forecast_length" not in st.session_state:
        st.session_state.forecast_length = 1
    if "altitude_max" not in st.session_state:
        st.session_state.altitude_max = 3000

    def date_controls():
        col1, col2, col3 = st.columns([0.2,0.6,0.2])

        with col1:
            if st.button("⏮️", use_container_width=True):
                st.session_state.forecast_date -= datetime.timedelta(days=1)
                st.rerun()
        with col2:
            #st.markdown(f"{st.session_state.forecast_date.strftime('%Y-%m-%d')}")
            st.session_state.forecast_date = st.date_input(
                "Start date", 
                value=st.session_state.forecast_date, 
                min_value=start_stop_time[0], 
                max_value=start_stop_time[1], 
                label_visibility="collapsed",
                disabled=True
                )
        with col3:
            if st.button("⏭️", use_container_width=True, disabled=(st.session_state.forecast_date == start_stop_time[1])):
                st.session_state.forecast_date += datetime.timedelta(days=1)
                st.rerun()
    
    time_start = datetime.time(0, 0)
    date_start = datetime.datetime.combine(st.session_state.forecast_date, time_start)
    date_end= datetime.datetime.combine(st.session_state.forecast_date+datetime.timedelta(days=st.session_state.forecast_length), datetime.time(0, 0))
    
    wind_fig = create_wind_map(
                subset,
                date_start=date_start, 
                date_end=date_end, 
                altitude_max=st.session_state.altitude_max)
    st.pyplot(wind_fig)
    date_controls()

    with st.expander("More settings", expanded=False):
        st.session_state.forecast_length = st.number_input("multiday", 1, 3, 1, step=1,)
        st.session_state.altitude_max = st.number_input("Max altitude", 0, 4000, 3000, step=500)
    
    ############################
    ######### SOUNDING #########
    ############################
    st.markdown("---")
    with st.expander("Sounding", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            sounding_date = st.date_input("Sounding date", value=now.date(), min_value=start_stop_time[0], max_value=start_stop_time[1])
        with col2:
            # set value to 1400
            sounding_time = st.time_input("Sounding time", value=datetime.time(14, 0))
        date = datetime.datetime.combine(sounding_date, sounding_time)


        with st.spinner('Building sounding...'):
            sounding_fig = create_sounding(subset, date=date.date(), hour=date.hour, altitude_max=st.session_state.altitude_max)
        st.pyplot(sounding_fig)

    st.markdown("Wind and sounding data from MEPS model (main model used by met.no). Thermal (green) is assuming ground temperature is 3 degrees higher than surrounding air. The location for both wind and sounding plot is Sogndal (61.22, 7.09). Ive probably made many errors in this process.")


if __name__ == "__main__":
    lat = 61.22908
    lon = 7.09674
    dataset_file_path = find_latest_meps_file()
    subset = load_meps_for_location(dataset_file_path, lat, lon, tol=0.1, altitude_min=0, altitude_max=3000)
    wind_fig = create_wind_map(subset, altitude_max=3000)
    sounding_fig = create_sounding(subset, date="2024-04-02", hour=15)