#%% 
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
px.set_mapbox_access_token(st.secrets['mapbox_token'])
import geopandas as gpd
import shapely.geometry
from shapely.affinity import affine_transform as T
from shapely.affinity import rotate as R
import streamlit.components.v1 as components
import utils

#%% Presentation
st.title("Flyinfo Sogn")
#%%
data = utils.get_weather_measurements()
#%%
import folium
import pandas as pd
from folium.features import DivIcon

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import streamlit as st
import streamlit_folium
import folium
import pandas as pd
from folium.features import DivIcon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
from folium.features import DivIcon
import matplotlib.colors as mcolors

import numpy as np
from matplotlib.colors import to_hex, LinearSegmentedColormap
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
        size = base_size + int((wind_speed / max_wind_speed) * base_size)

        # Interpolate color based on wind speed
        color = interpolate_color(wind_speed)

        # Adjust angle for map orientation
        adjusted_angle = angle + 90

        # Define font size for text
        text_font_size = 10

        # Adding text for wind speed next to the arrow
        arrow_html = f'<div style="display: flex; align-items: center;">'
        arrow_html += f'<div style="font-size: {size}px; color: {color}; transform: rotate({adjusted_angle}deg);">&#10148;</div>'
        arrow_html += f'<span style="margin-left: 5px; font-size: {text_font_size}px;">{wind_speed}m/s'
        if wind_gust:
            arrow_html += f'({wind_gust})'
        arrow_html += '</span></div>'

        return DivIcon(icon_size=(size * 2, size), icon_anchor=(size, size // 2), html=arrow_html)

    # Create a Folium map
    m = folium.Map(location=[61.1732881, 7.1195861], zoom_start=8)

    # Add wind direction arrows to the map
    for station_name, station in data.items():
        latest_measurement = station['measurements'].iloc[0]
        wind_speed = latest_measurement['wind_strength']
        wind_angle = latest_measurement['wind_angle']
        wind_gust = latest_measurement.get('wind_gust')

        # Create an arrow icon
        arrow_icon = create_arrow_icon(wind_speed, wind_gust, wind_angle)

        # Add the arrow to the map
        folium.Marker(
            [station['lat'], station['lon']],
            icon=arrow_icon,
            popup=f"<b>{station_name}</b><br>Speed: {wind_speed} m/s"
        ).add_to(m)

    return folium_static(m)

import streamlit as st
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import pandas as pd

# Function to create wind rose
def plot_wind_rose(df, rmax=20):
    #plt.style.use('seaborn')  # Use seaborn style for better visuals
    fig = plt.figure()
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(df['wind_angle'], df['wind_strength'], normed=True, opening=0.8, edgecolor='white')
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
        max_frequency = max(max_frequency, filtered_df['wind_strength'].value_counts().max())


    # Calculate the number of columns based on screen width
    cols = st.columns(2)# if st.session_state.window_width > 768 else st.beta_columns(1)

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

if __name__ == "__main__":
    build_live_map(data)
    wind_rose(data)

# %%
