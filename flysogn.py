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
from windrose import WindroseAxes
import collect_ogn
import datetime

# Set correct timezone
import os, time
os.environ['TZ'] = 'CET'
time.tzset()

def build_live_map(data):
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
    # Create a Folium map
    m = folium.Map(location=[61.1732881, 7.1195861], zoom_start=9, width='100%', height='100%')

    # Add wind direction arrows to the map
    for station_name, station in data.items():
        latest_measurement = station['measurements'].iloc[0]
        wind_speed = latest_measurement['wind_speed']
        wind_direction = latest_measurement['wind_direction']
        wind_gust = latest_measurement.get('wind_gust')

        # Create an arrow icon
        arrow_icon = create_arrow_icon(wind_speed, wind_gust, wind_direction)

        # Add the arrow to the map
        folium.Marker(
            [station['lat'], station['lon']],
            icon=arrow_icon,
            popup=f"<b>{station_name}</b> {wind_speed} ({wind_gust if wind_gust else 'N/A'}) m/s (gust)"
        ).add_to(m)

    # Display latest positions of aircraft
    for aircraft, pos in st.latest_pos.items():
        # only plot aircrafts updated the last 1 hrs:
        #print((datetime.datetime.now(datetime.timezone.utc) - pos['timestamp']).seconds )
        if (datetime.datetime.now() - pos['timestamp']).seconds < 3600*4:
            folium.Marker(
                [pos['latitude'], pos['longitude']],
                icon=folium.Icon(color='blue', icon='plane', angle=30), # color='blue',
                popup=f"<b>{aircraft}</b> altitude: {pos['altitude']:.0f} mas timestamp: {pos['timestamp']}"
            ).add_to(m)

    return folium_static(m)

# Function to create wind rose
def plot_wind_rose(df, rmax=20):
    #plt.style.use('seaborn')  # Use seaborn style for better visuals
    fig = plt.figure()
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(df['wind_direction'], df['wind_speed'], normed=True, opening=0.8, edgecolor='white')
    ax.set_rmax(rmax)
    return fig

# Streamlit app
def wind_rose(data):
    # User input for lookback period
    lookback_hours = st.slider("Select lookback period in hours", 1, 48, 12)

    max_frequency = 0
    for station_info in data.values():
        df = station_info['measurements']
        df['time'] = pd.to_datetime(df['time'])
        min_time = df['time'].max() - pd.Timedelta(hours=lookback_hours)
        filtered_df = df[df['time'] >= min_time]
        max_frequency = max(max_frequency, filtered_df['wind_speed'].value_counts().max())


    # Calculate the number of columns based on screen width
    cols = st.columns(2)

    # Process and display wind roses for each station
    idx = 0  # Index to track current column
    for station_name, station_info in data.items():
        with cols[idx % len(cols)]:
            st.subheader(f"Wind Rose for {station_name}")

            # Convert 'time' column to datetime and filter based on lookback period
            df = station_info['measurements']
            df['time'] = pd.to_datetime(df['time'])
            min_time = df['time'].max() - pd.Timedelta(hours=lookback_hours)
            filtered_df = df[df['time'] >= min_time]

            # Display wind rose
            st.pyplot(plot_wind_rose(filtered_df, rmax=max_frequency))

        idx += 1

def plot_wind_data(df_dict, selected_stations, data_type, yaxis_title, lookback_hours):
    fig = go.Figure()

    # Define a list of colors
    colors = ['rgba(255, 0, 0, 0.8)', 'rgba(0, 255, 0, 0.8)', 'rgba(0, 0, 255, 0.8)', 'rgba(255, 255, 0, 0.8)', 'rgba(0, 255, 255, 0.8)', 'rgba(255, 0, 255, 0.8)']

    for i, station in enumerate(selected_stations):
        df = df_dict[station]['measurements']
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df.set_index('time', inplace=True)

        # Resample and interpolate data to be every 15 minutes
        df = df.resample('15T').interpolate()

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

def historical_wind_graphs(data):
    # Lookback period slider
    # Add columns for this
    col1, col2 = st.columns(2)
    with col1:
        lookback_hours = st.slider("Select lookback period in hours", 1, 48, 10)
    
    with col2:
        default_station_historical = ["Barten", "Modvaberget", "Tylderingen"]
        selected_stations = st.multiselect("Select Stations", options=list(data.keys()), default=default_station_historical)

    if selected_stations:
        # Filter data based on lookback period and plot
        for data_type, yaxis_title in [('wind_speed', 'Wind Strength (m/s)'), ('wind_direction', 'Wind Angle (degrees)')]:
            fig = plot_wind_data(data, selected_stations, data_type, yaxis_title, lookback_hours)
            st.plotly_chart(fig, use_container_width=True)

def show_webcams():
    images = ["http://sognskisenter.org/webkam/parkering/image.jpg",
    "http://sognskisenter.org/webkam/rodekorshytta/image.jpg",
    "http://sognskisenter.org/webkam/mast16/image.jpg"]
    st.image(images, use_column_width=True)

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

if __name__ == "__main__":
    st.set_page_config(page_title="Flysogn",page_icon="🪂", layout="wide")

    # start ogn collector
    if not hasattr(st, 'client_started'):
        collect_ogn.start_client()
        st.client_started = True

    if not hasattr(st, 'data'):
        with st.spinner('Wait for it...'):
            st.weather_data = utils.get_weather_measurements()

    # Create tabs
    tab_livemap, tab_history, tab_livetrack, tab_windrose, tab_webcam, tab_windy, tab_holfuy, live_pilot_list = st.tabs(["Live map", "Historical weather", "livetrack", "Wind rose","Webcams","Windy", "holfuy", "Live pilot list"])

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
        build_live_map(st.weather_data)
    with live_pilot_list:
        st.subheader("Latest aircraft positions")
        st.dataframe(st.latest_pos)
    with tab_history:
        historical_wind_graphs(st.weather_data)
    with tab_livetrack:
        show_puretrack()
    with tab_windrose:
        wind_rose(st.weather_data)
    with tab_webcam:
        show_webcams()
    with tab_windy:
        show_windy()
    with tab_holfuy:
        show_holfuy_widgets()
    #with tab_forecast:
    #    vestavind.show_forecast()

    st.title("Flyinfo Sogn")
    st.text("Information gathered from various sources. Hobby project because why not.")

    # Update live weather data in the background
    st.weather_data = utils.get_weather_measurements()