import folium
from streamlit_folium import st_folium
import streamlit as st
with st.expander("Map", expanded=True):
    @st.cache_data(ttl=30)
    def build_map():
        m = folium.Map(location=[61.22908, 7.09674], zoom_start=9, tiles="openstreetmap")
        m.add_child(folium.LatLngPopup())

        marker = folium.Marker(location=[61.22908, 7.09674], popup='Marker Popup Text')
        m.add_child(marker)
        folium.plugins.Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)
        return m
    m = build_map()
    map = st_folium(m,width="50%")

    if map['last_clicked'] is not None:
        st.write(map['last_clicked']['lat'],map['last_clicked']['lng'])
    