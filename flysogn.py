#%% 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import utils
import folium
from folium.features import DivIcon
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
from matplotlib.colors import to_hex, LinearSegmentedColormap
import collect_ogn
import datetime
import altair as alt
import polars as pl
# Set correct timezone
import os, time
os.environ['TZ'] = 'CET'
time.tzset()

def create_arrow_icon(wind_speed, wind_gust, angle, max_wind_speed=20):
    # Adjust the size of the arrow based on wind speed
    base_size = 20
    try:
        size = base_size + int((wind_speed / max_wind_speed) * base_size)
    except:
        size = base_size

    # Interpolate color based on wind speed
    color = interpolate_color(wind_speed)

    # Adjust angle for map orientation
    adjusted_angle = angle + 90

    # Define font size for text
    text_font_size = 10

    # Adding text for wind speed next to the arrow
    arrow_html = f'<div style="display: flex; align-items: center;">'
    arrow_html += f'<div style="font-size: {size}px; color: {color}; transform: rotate({adjusted_angle}deg); text-shadow: -1px 0 black, 0 1px black, 1px 0 black, 0 -1px black;">&#10148;</div>'
    arrow_html += f'<span style="margin-left: 5px; font-size: {text_font_size}px;">{wind_speed}m/s'
    if wind_gust:
        arrow_html += f'({wind_gust})'
    arrow_html += '</span></div>'

    return DivIcon(icon_size=(size * 2, size), icon_anchor=(size, size // 2), html=arrow_html)

def create_wind_chart(wind_chart_data, station_name):
    now = pd.Timestamp(datetime.datetime.now(), tz='CET')
    wind_chart_data = wind_chart_data[wind_chart_data['time'] >= now - pd.Timedelta(hours=6)]
    wind_chart_data = wind_chart_data.groupby('time', as_index=False).first()
    if wind_chart_data.empty:
        return None


    wind_chart_data = (
        wind_chart_data
        .set_index('time')
        .resample('15min')
        .ffill()
        .infer_objects(copy=False)
        .interpolate()
        .reset_index(drop=False)
    )

    # Calculate min and max time for the full 24-hour span
    min_time = wind_chart_data['time'].min()
    max_time = wind_chart_data['time'].max()

    # Area 1: Green (0 m/s to 5 m/s)
    green_area = alt.Chart(pd.DataFrame({'time': [min_time, max_time], 'y0': 0, 'y1': 5})).mark_area(
        color='rgba(0, 255, 0, 0.5)'  # Transparent green
    ).encode(
        x='time:T',
        y='y0:Q',
        y2='y1:Q'
    )

    # Area 2: Yellow (5 m/s to 10 m/s)
    yellow_area = alt.Chart(pd.DataFrame({'time': [min_time, max_time],'y0' : 5, 'y1' : 10})).mark_area(
        color='rgba(255, 255, 0, 0.5)'  # Transparent yellow
    ).encode(
        x='time:T',
        y='y0:Q',
        y2='y1:Q'
    )

    # Area 3: Black (10 m/s and above)
    max_val = max(10,wind_chart_data.get("wind_gust",np.zeros(1)).max(),wind_chart_data['wind_speed'].max())
    black_area = alt.Chart(pd.DataFrame({'time': [min_time, max_time], 'y0': 10, 'y1': max_val})).mark_area(
        color='rgba(0, 0, 0, 0.5)'  # Transparent black
    ).encode(
        x='time:T',
        y='y0:Q',
        y2='y1:Q'
    )

    wind_chart_data['wind_direction'] = wind_chart_data['wind_direction'] % 360

    # Add wind direction arrows
    wind_direction_arrows = (
        alt.Chart(wind_chart_data)
        .transform_calculate(
            angle="180 + datum.wind_direction"  # Compute rotation angle
        )
        .mark_text(
            text='➤',  # Unicode text for the arrow
            align='center',
            baseline='middle',
            fontSize=12
        )
        .encode(
            x='time:T',
            y=alt.value(12),  # Fixed position for separation
            angle=alt.Angle('angle:Q'),  # Use calculated angle
            tooltip=['time:T', 'wind_direction:Q']
        )
        .transform_filter(
            alt.datum.wind_direction != None
        )
    )

    # Define two line charts
    wind_speed_chart = alt.Chart(wind_chart_data).mark_line(color='blue').encode(
        x='time:T',
        y=alt.Y('wind_speed:Q', title='Wind Speed (m/s)'),
        tooltip=['time:T', 'wind_speed:Q']
    )

    wind_gust_chart = alt.Chart(wind_chart_data).mark_line(strokeDash=[5, 5], color='red').encode(
        x='time:T',
        y=alt.Y('wind_gust:Q', title='Wind Gust (m/s)'),
        tooltip=['time:T', 'wind_gust:Q']
    ).transform_filter(
        alt.datum.wind_gust != None
    )

    # Combine charts
    chart = alt.layer(green_area, yellow_area, black_area, wind_speed_chart, wind_gust_chart, wind_direction_arrows).properties(
        width=300,
        height=150,
        title=f"{station_name} Wind (last 24 hours)"
    ).interactive()

    return chart.to_dict()

def interpolate_color(wind_speed, thresholds=[2, 8, 14], colors=['white', 'green', 'red', 'black']):
    # Normalize thresholds to range [0, 1]
    norm_thresholds = [t / max(thresholds) for t in thresholds]
    norm_thresholds = [0] + norm_thresholds + [1]

    # Extend color list to match normalized thresholds
    extended_colors = [colors[0]] + colors + [colors[-1]]

    # Create colormap
    cmap = LinearSegmentedColormap.from_list("wind_speed_cmap", list(zip(norm_thresholds, extended_colors)), N=256)

    # Normalize wind speed to range [0, 1] and get color
    norm_wind_speed = wind_speed / max(thresholds)
    return to_hex(cmap(np.clip(norm_wind_speed, 0, 1)))

def build_live_map(stations, measurements):
    """
    Build a live map using st.stations and st.measurements.
    
    Parameters:
        stations (list[dict]): A list (or dict) of station information. Each station dict 
                               should contain at least 'name', 'lat', and 'lon'.
        measurements (pl.DataFrame): A Polars DataFrame with weather measurements having 
                                     a 'name' column that matches stations.
        
    Returns:
        folium_static map object.
    """
    # Create a Folium map
    m = folium.Map(location=[61.26, 7.1195861], zoom_start=10, width='100%', height='100%')
    
    # Add webcam markers using a smaller DivIcon so that they're in the background
    webcams = [
        {
            'name': "Sogn skisenter parkering",
            'url': "http://sognskisenter.org/webkam/parkering/image.jpg",
            'latitude': 61.335706,
            'longitude': 7.217362,
        },
        {
            'name': "Rødekorshytta",
            'url': "http://sognskisenter.org/webkam/rodekorshytta/image.jpg",
            'latitude': 61.342406,
            'longitude': 7.184607,
        },
        {
            'name': "Sogn skisenter Mast 16",
            'url': "http://sognskisenter.org/webkam/mast16/image.jpg",
            'latitude': 61.339414,
            'longitude': 7.193114,
        },
        {
            'name': "Rindabotn",
            'url': "https://cdn.norwaylive.tv/snapshots/6637b019-aeab-4a45-b671-f1f9bae39d09/kam1utsnitt2.jpg",
            'latitude': 61.289154,
            'longitude': 6.967829
        },
        {
            'name': "Turtagrø",
            'url': "https://turtagro.no/images/image_00001.jpg",
            'latitude': 61.5043928,
            'longitude': 7.8012656
        }
    ]
    for webcam in webcams:
        webcam_icon = folium.DivIcon(
            html='<div style="font-size:12px; color:green;">📷</div>',
            icon_size=(24, 24),
            icon_anchor=(12, 12)
        )
        folium.Marker(
            [webcam['latitude'], webcam['longitude']],
            icon=webcam_icon,
            popup=folium.Popup(f'<img src="{webcam["url"]}" alt="Webcam Image" width="300">', max_width=300)
        ).add_to(m)
    
    # Add station markers with weather arrows above the webcam markers 
    for station in stations:
        # Filter measurements for this station using polars filtering then convert to pandas
        station_measurements = (
            measurements.filter(pl.col('name') == station['name'])
                        .to_pandas()
        )
        # Skip if there are no measurements or if wind_direction is missing
        if station_measurements.empty or 'wind_direction' not in station_measurements.columns:
            continue
        
        latest_measurement = station_measurements.iloc[0]
        wind_speed = latest_measurement['wind_speed']
        wind_direction = latest_measurement['wind_direction']
        
        # If available, get wind_gust; otherwise, None
        wind_gust = latest_measurement.get('wind_gust', None)
        if np.isnan(wind_speed):
            continue

        # Create an arrow icon based on the latest measurement
        arrow_icon = create_arrow_icon(wind_speed, wind_gust, wind_direction)

        # Prepare time-series wind data for Vega chart
        wind_chart = create_wind_chart(station_measurements, station['name'])
        if wind_chart is None:
            continue

        # Add the arrow marker with a popup containing the wind chart
        folium.Marker(
            [station['lat'], station['lon']],
            icon=arrow_icon,
            popup=folium.Popup(max_width=460).add_child(folium.VegaLite(wind_chart))
        ).add_to(m)

    # Add aircraft markers (unchanged)
    for aircraft, pos in st.latest_pos.items():
        # Plot only aircraft updated the last 4 hours
        if (datetime.datetime.now() - pos['timestamp']).seconds < 3600 * 4:
            pg_icon = folium.CustomIcon(
                'pgicon.png',  # Replace with the path to your custom icon
                icon_size=(20, 20)
            )
            plane_icon = folium.CustomIcon(
                'planeicon.png',  # Replace with the path to your custom icon
                icon_size=(20, 20)
            )
            icon = pg_icon if "aircraft" in pos.get('beacon_type', '') else plane_icon
            folium.Marker(
                [pos['latitude'], pos['longitude']],
                icon=icon,
                popup=f"<b>{aircraft}</b> altitude: {pos['altitude']:.0f} mas timestamp: {pos['timestamp']}"
            ).add_to(m)

    return folium_static(m)

def plot_wind_data(df_dict, selected_stations, data_type, yaxis_title, lookback_hours):
    fig = go.Figure()

    # Define a list of colors
    colors = ['rgba(255, 0, 0, 0.8)', 'rgba(0, 255, 0, 0.8)', 'rgba(0, 0, 255, 0.8)', 'rgba(255, 255, 0, 0.8)', 'rgba(0, 255, 255, 0.8)', 'rgba(255, 0, 255, 0.8)']

    for i, station in enumerate(selected_stations):
        df = df_dict[station]['measurements']
        df = df.sort_values('time')
        df.set_index('time', inplace=True)

        # Resample and interpolate data to be every 15 minutes
        df = (
            df
            .infer_objects(copy=False)
            .resample('15min')
            .interpolate()
        )

        # Filter data based on lookback period
        min_time = df.index.max() - pd.Timedelta(hours=lookback_hours)
        filtered_df = df[df.index >= min_time]

        if data_type in filtered_df.columns:
            fig.add_trace(go.Scatter(
                x=filtered_df.index, 
                y=filtered_df[data_type], 
                mode='markers' if data_type == 'wind_direction' else 'lines+markers',  # Change mode to 'markers' for wind angle
                name=station,
                line=dict(width=2, color=colors[i % len(colors)]),  # Use color from colors list
                marker=dict(size=5),  # Adjust marker size here
                legendgroup=station,  # Assign legend group
            ))

        # Add wind gust data as a scatter plot with markers only for wind strength
        if 'wind_gust' in filtered_df.columns and data_type == 'wind_speed':
            fig.add_trace(go.Scatter(
                x=filtered_df.index, 
                y=filtered_df['wind_gust'], 
                mode='markers',  # Change mode to 'markers'
                name=f"{station} gust",
                marker=dict(size=5, color=colors[i % len(colors)]),  # Use color from colors list
                legendgroup=station,  # Assign same legend group as wind plot
                showlegend=False,  # Hide legend entry
            ))


    # Update layout
    fig.update_layout(
        title_text=f"{yaxis_title}",
        xaxis=dict(
            title='Time', 
            title_font_size=14,
            tickformat='%Y-%m-%d %H:%M',
            gridcolor='grey',  # Change grid line color if needed
            showgrid=True,
            griddash='dash',  # Set grid line style
            dtick=3600000 * 1  # 1 hours in milliseconds
        ),
        yaxis=dict(
            title=yaxis_title, 
            title_font_size=14
        ),
        margin=dict(l=10, r=10, t=30, b=40),
        title_font_size=16,
        legend=dict(
            y=-0.1,  # Adjust this value to move the legend up or down
            x=0.5,  # Adjust this value to move the legend left or right
            xanchor="center",  # Anchor the legend at its center
            orientation="h"  # Horizontal orientation
        )
    )

    return fig

def historical_wind_graphs(stations, measurements):
    # Create two columns for the sliders and multi-select widget.
    col1, col2 = st.columns(2)
    with col1:
        lookback_hours = st.slider("Select lookback period in hours", 1, 48, 10)
    with col2:
        # Default station names for historical view.
        default_station_historical = ["Barten", "Modvaberget", "Tylderingen"]
        # Build a list of available station names from st.stations.
        station_names = [station['name'] for station in stations]
        selected_stations = st.multiselect("Select Stations", options=station_names, default=default_station_historical)

    if selected_stations:
        # Build a dictionary mapping each selected station name to its measurements.
        measurement_by_station = {}
        for station in selected_stations:
            # Filter measurements for this station, convert to pandas DataFrame and drop duplicate times.
            station_measurements = measurements.filter(pl.col("name") == station).to_pandas()
            station_measurements.drop_duplicates(subset=['time'], inplace=True)
            measurement_by_station[station] = {'measurements': station_measurements}

        # Plot charts for different data types.
        for data_type, yaxis_title in [('wind_speed', 'Wind Strength (m/s)'), ('wind_direction', 'Wind Angle (degrees)')]:
            fig = plot_wind_data(measurement_by_station, selected_stations, data_type, yaxis_title, lookback_hours)
            st.plotly_chart(fig, use_container_width=True)

def show_windy():
    st.components.v1.iframe(
        src="https://embed.windy.com/embed2.html?lat=61.010&lon=7.015&detailLat=61.249&detailLon=7.086&width=650&height=450&zoom=8&level=850h&overlay=wind&product=ecmwf&menu=&message=true&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=true&metricWind=m%2Fs&metricTemp=%C2%B0C&radarRange=-1",
        height=450
    )

def show_holfuy_widgets():
    for stationId in [1550,1703]:
        components.iframe(
            src=f"https://widget.holfuy.com/?station={stationId}&su=m/s&t=C&lang=en&mode=detailed",
            height=250
        )
        components.iframe(
            src=f"https://widget.holfuy.com/?station={stationId}&su=m/s&t=C&lang=en&mode=average&avgrows=32",
            height=170
    )
        
def show_puretrack():
    url = "https://www.burnair.cloud/?layers=%2Clz%2Clzp%2Cto%2Ctodhv%2Clp%2Cpz%2Cfp%2Cna%2Cle%2Cca%2Cvw%2Ccc%2Ctt%2Cpt%2Ctl%2Ctp%2Cpp%2Cmp%2Cw-ch-uw%2Cc_20&visibility=%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Coff%2Cauto&base=bbt#12/61.2402/7.1298"

    url = "https://puretrack.io/?l=61.34928,7.13707&z=9.6"
    components.iframe(url, height=600)


def plot_sounding(stations, measurements):
    """
    Build a sounding plot from stations and measurements using Polars.

    Parameters:
      stations (list[dict]): List of station info dictionaries. Each station must have
                             at least a 'name' and an altitude (provided as 'altitude' or 'elevation').
      measurements (pl.DataFrame): A Polars DataFrame containing weather measurements.
                                   It must have a 'name' column to match stations and a 'time' column.
                                   
    Renders:
      A matplotlib plot via st.pyplot.
    """
    import matplotlib.pyplot as plt
    import scipy.interpolate
    import numpy as np
    import polars as pl
    import datetime
    import streamlit as st


    # Get the overall minimum and maximum times from the measurements
    min_time = measurements.select(pl.min("time")).item()
    max_time = measurements.select(pl.max("time")).item()

    # Let user select a datetime using the slider. Convert to python datetime.
    selected_datetime = st.slider(
        "Select datetime",
        min_value=min_time,
        max_value=max_time,
        value=max_time,
        format="YYYY-MM-DD HH:mm:ss",
        step=datetime.timedelta(minutes=15),
    )

    temperatures = []
    altitudes = []
    station_names = []

    # Loop over each station
    for station in stations:
        altitude = station.get('altitude') or station.get('elevation')
        if altitude is None:
            continue

        # Filter measurements for this station by matching the 'name' field.
        df = measurements.filter(pl.col("name") == station["name"])
        if df.is_empty() or "temperature" not in df.columns:
            continue

        # Compute absolute difference between each measurement time and selected time.
        df = df.with_columns(
            time_diff=(pl.col("time")-pl.lit(selected_datetime))
        )

        # Get the row with the smallest time_diff.
        closest = df.sort("time_diff").head(1)
        temp = closest.select("temperature").item()
        if temp is None or (isinstance(temp, float) and np.isnan(temp)):
            continue

        temperatures.append(temp)
        altitudes.append(altitude)
        station_names.append(station["name"])

    if len(temperatures) == 0:
        st.write("No temperature data available for the selected time.")
        return

    # Convert lists to arrays and sort by altitude.
    temperatures = np.array(temperatures)
    altitudes = np.array(altitudes)
    station_names = np.array(station_names)
    
    sorted_indices = np.argsort(altitudes)
    altitudes = altitudes[sorted_indices]
    temperatures = temperatures[sorted_indices]
    station_names = station_names[sorted_indices]

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(temperatures, altitudes, 'o', label='Observed Temperature', markerfacecolor='blue', markersize=8)
    
    for temp, alt, name in zip(temperatures, altitudes, station_names):
        ax.annotate(name, (temp, alt), textcoords="offset points", xytext=(5,5), ha='left')
    
    if len(temperatures) > 2:
        spline = scipy.interpolate.UnivariateSpline(altitudes, temperatures, k=1, s=5)
        altitude_smooth = np.linspace(altitudes[0], altitudes[-1], 50)
        temperature_smooth = spline(altitude_smooth)
        ax.plot(temperature_smooth, altitude_smooth, '-', color='red', label='Smoothed Temperature Curve')
    
    Gamma_d = 0.0098  # Dry Adiabatic Lapse Rate in °C/m
    Altitude_dalr = np.linspace(0, altitudes[-1], 10)
    Temperature_dalr = 0 - Gamma_d * (Altitude_dalr - 0)
    
    x_min, x_max = ax.get_xlim()
    min_offset = min(0, x_min)
    max_offset = max(10, x_max + 10)
    offsets = np.arange(min_offset, max_offset, 1)
    for offset in offsets:
        ax.plot(Temperature_dalr + offset, Altitude_dalr, '--', color='grey', alpha=0.3)
    ax.plot([], [], '--', color='grey', alpha=0.5, label='Dry Adiabatic Lapse Rate')
    
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Sounding at {selected_datetime}')
    ax.grid(True)
    ax.legend()
    
    st.pyplot(fig)



if __name__ == "__main__":
    st.set_page_config(
        page_title="Flysogn",
        page_icon="🪂", 
        layout="wide",
        menu_items={
            #'Get Help': 'Hah, you thought you could get help here?',
            'Report a bug': "https://github.com/simeneide/flysogn/issues",
            'About': "Made by Simen Eide on his spare time when he should have been out in the sun."
            }
        )
    if not hasattr(st, 'data'):
        with st.spinner('Wait for it...'):
            st.stations, st.measurements = utils.get_weather_measurements()

    # start ogn collector
    if not hasattr(st, 'client_started'):
        collect_ogn.start_client()
        st.client_started = True



    # Create tabs
    tab_livemap,tab_sounding, tab_history, tab_livetrack, tab_windy, tab_holfuy, live_pilot_list = st.tabs(["Live map", "Live sounding", "Historical weather", "Puretrack","Windy", "holfuy", "Live pilot list"])

    # Make folio map width response:
    # https://github.com/gee-community/geemap/issues/713
    st.markdown("""
        <style>
        iframe {
            width: 100%;
            min-height: 200px;
            height: 100%:
        }
        </style>
        """, unsafe_allow_html=True)

    # Content for the first tab
    with tab_livemap:
        build_live_map(st.stations, st.measurements)
    with tab_sounding:
        plot_sounding(st.stations, st.measurements)
    
    with live_pilot_list:
        st.subheader("Latest aircraft positions")
        st.dataframe(st.latest_pos)
    with tab_history:
        historical_wind_graphs(st.stations, st.measurements)
    with tab_livetrack:
        show_puretrack()
    with tab_windy:
        show_windy()
    with tab_holfuy:
        show_holfuy_widgets()

    
    
    url = "https://www.xcontest.org/world/en/flights-search/?list[sort]=pts&filter[point]=7.103548%2061.345346&filter[radius]=45196&filter[mode]=CROSS&filter[date_mode]=dmy&filter[date]=&filter[value_mode]=dst&filter[min_value_dst]=&filter[catg]=&filter[route_types]=&filter[avg]=&filter[pilot]="
    st.markdown(f"""
## Paragliding info for Sogn og Fjordane, Norway
Information gathered from various sources. Hobby project because why not.
Sogn is in the end of the world's longest ice free fjord and a great place to fly when the weather is good. Our xc season starts in April and ends in September, with the best conditions early on when the snow has melted near the fjord, and stays in the mountains. Check out flights on [XContest]({url}), or check out a timelapse flight from [Sogndal]('https://www.youtube.com/watch?v=YtigYp1HFzk').
                
Site is "best effort maintained" by [Simen Eide](https://www.instagram.com/simenfly/), code is available on [Github](https://github.com/simeneide/flysogn).""")
    
    # Update live weather data in the background
    utils.write_weather_measurements_to_db(lookback=72)