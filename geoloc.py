import streamlit as st
import pydeck as pdk
from streamlit_js_eval import get_geolocation
import random


def update_locations():
    # Initialize session state for storing user_locations
    if 'user_locations' not in st.session_state:
        st.session_state.user_locations = []

    # get current location    
    location = get_geolocation()
    # jitter location randomly
    location['coords']['latitude'] += random.uniform(-0.001, 0.001)
    location['coords']['longitude'] += random.uniform(-0.001, 0.001)
    
    st.session_state.user_locations.append([location['coords']['longitude'], location['coords']['latitude']])
    st.write(st.session_state.user_locations)
    return location
# Get current location
location = update_locations()

# Set the map's initial view
initial_view_state = pdk.ViewState(
    latitude=location['coords']['latitude'] if location else 0,
    longitude=location['coords']['longitude'] if location else 0,
    zoom=11,
    pitch=0,
)

# Create a layer to display on the map
layer = pdk.Layer(
    'ScatterplotLayer',
    st.session_state.user_locations,
    get_position='[0, 1]',
    get_color='[200, 30, 0, 160]',
    get_radius=100,
)

# Render the map
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=initial_view_state,
    layers=[layer],
))

# Button to update location manually (optional)
#if st.button('Update My Location'):
#    st.rerun()