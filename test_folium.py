import folium
from streamlit_folium import st_folium
import streamlit as st
m = folium.Map(location=[61.22908, 7.09674], zoom_start=9, tiles="openstreetmap")
m.add_child(folium.LatLngPopup())

marker = folium.Marker(location=[61.22908, 7.09674], popup='Marker Popup Text')
m.add_child(marker)

map = st_folium(m,width="50%")

if map['last_clicked'] is not None:
    st.write(map['last_clicked']['lat'],map['last_clicked']['lng'])