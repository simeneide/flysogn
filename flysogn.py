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



#%%

#%% Presentation
st.title("Flyinfo Sogn")
#%%
st.subheader("Barten")
df_barten = utils.get_historical_ecowitt()
fig = go.Figure(data=go.Scatter(x=df_barten.time, y=df_barten.wind_strength, name="Wind Strength"))
# Add gusts to figure
fig.add_trace(go.Scatter(x=df_barten.time, y=df_barten.wind_gust, name="Gust"))

fig.update_layout(title="Vindstyrke Barten", xaxis_title="Tid", yaxis_title="Vindstykr [m/s]")
st.plotly_chart(fig)

fig = go.Figure(data=go.Scatter(x=df_barten.time, y=df_barten.wind_angle))
fig.update_layout(title="Vindretning Barten", xaxis_title="Tid", yaxis_title="Vindretning [°]")
st.plotly_chart(fig)

#%% Modvoberget
components.iframe(
    src="https://widget.holfuy.com/?station=1550&su=m/s&t=C&lang=en&mode=detailed",
    height=250
)
components.iframe(
    src="https://widget.holfuy.com/?station=1550&su=m/s&t=C&lang=en&mode=average&avgrows=32",
    height=170
)
st.subheader("Storhogen")
try:
    
    df_storhogen = utils.get_storhogen_data()
    last_obs = df_storhogen.tail(1).to_dict("records")[0]
    
    st.markdown(f"""
    **time:** \t {last_obs['time']}  
    **Hours since reading**: \t {last_obs['hours_since_reading']:.1f}  
    speed:  \t {last_obs['wind_strength']:.1f} m/s  
    direction: \t {last_obs['wind_angle']}°
    """)
except:
    pass

#%%
st.subheader("Live map")
def plot_wind_arrows(df):
    a = shapely.wkt.loads(
        "POLYGON ((-0.6227064947841563 1.890841205238906, -0.3426264166591566 2.156169330238906, -0.07960493228415656 2.129731830238906, 1.952059130215843 0.022985736488906, -0.2085619635341561 -2.182924419761094, -0.6397611822841562 -1.872877544761094, -0.6636088385341563 -1.606053326011095, 0.5862935052158434 -0.400158794761094, -2.312440869784157 -0.3993228572610942, -2.526870557284156 -0.1848931697610945, -2.517313916659156 0.2315384708639062, -2.312440869784157 0.3990052677389059, 0.5862935052158434 0.399841205238906, -0.6363314947841564 1.565763080238906, -0.6227064947841563 1.890841205238906))"
    )
    # scatter points
    t = (
        px.scatter_mapbox(df, lat="lat", lon="lon", 
        color="wind_strength", hover_data=["wind_strength","wind_angle","hours_since_reading","name","altitude"])
        .update_layout(mapbox={"style": "carto-positron"})
        .data
    )

    # wind direction and strength
    fig = px.choropleth_mapbox(
        df,
        geojson=gpd.GeoSeries(
            df.loc[:, ["lon", "lat", "wind_angle", "s"]].apply(
                lambda r: R(
                    T(a, [r["s"], 0, 0, r["s"], r["lon"], r["lat"]]),
                    angle=-90-r['wind_angle'],
                    origin=(r["lon"], r["lat"]),
                    use_radians=False,
                ),
                axis=1,
            )
        ).__geo_interface__,
        locations=df.index,
        color="wind_strength",
        opacity=0.5
    )
    fig.add_traces(t)
    fig.update_layout(
        mapbox={
            "style": "carto-positron", 
            "zoom": 8,
                    "center":dict(
                lat=df['lat'].mean(),
                lon=df['lon'].mean()
            )}, margin={"l":0,"r":0,"t":0,"b":0})
    return fig
#df = collect_netatmo_data()
holfuy = utils.collect_holfuy_data() # Append modvaberget
storhogen = df_storhogen.tail(1)
storhogen['s'] = np.sqrt(storhogen['wind_strength'])/200
df = pd.concat([holfuy, storhogen.tail(1),df_barten.tail(1)]) #  df
fig = plot_wind_arrows(df)
fig
#%% WINDY
st.subheader("Windy nå 1500moh")
st.components.v1.iframe(
    src="https://embed.windy.com/embed2.html?lat=61.010&lon=7.015&detailLat=61.249&detailLon=7.086&width=650&height=450&zoom=8&level=850h&overlay=wind&product=ecmwf&menu=&message=true&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=true&metricWind=m%2Fs&metricTemp=%C2%B0C&radarRange=-1",
    height=450
)

#%% Vindnå.no
components.iframe(
    src="https://vindnå.no",
    height=450
)

#%% IMAGES
images = ["http://sognskisenter.org/webkam/parkering/image.jpg",
"http://sognskisenter.org/webkam/rodekorshytta/image.jpg",
"http://sognskisenter.org/webkam/mast16/image.jpg"]
st.image(images, use_column_width=True)

#%% Historical data
fig = go.Figure(data=go.Scatter(x=df_storhogen.time, y=df_storhogen.wind_angle))
fig.update_layout(title="Vindretning storhogen", xaxis_title="Tid", yaxis_title="Vindretning [°]")
st.plotly_chart(fig)

fig = go.Figure(data=go.Scatter(x=df_storhogen.time, y=df_storhogen.wind_strength))
fig.update_layout(title="Vindstyrke storhogen", xaxis_title="Tid", yaxis_title="Vind [m/s]", autosize=True)
st.plotly_chart(fig)


st.text("""
Visualisering av metar-data fra storhogen for lettere å få en oversikt over vindforhold akkurat nå.
Repo: https://github.com/simeneide/flysogn
Data er hentet fra api.met.no
""")




"""
#%% Text forecast
from lxml import etree as ET
import requests
r = requests.get("https://api.met.no/weatherapi/textforecast/2.0/landoverview")
#import xml
#import xml.etree.ElementTree as ET
et = ET.fromstring(r.content)
for child in et.findall("time"):
    print(child.findall("*"))
fc = et.xpath(".//time")[0].xpath(".//location")[0]

fc.items()
# %%
"""