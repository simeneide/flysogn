#%% 
from http import client
from numpy.core.numeric import Inf
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from metar import Metar
import json
import plotly.graph_objects as go
import plotly.express as px
px.set_mapbox_access_token(st.secrets['mapbox_token'])
import geopandas as gpd
import shapely.geometry
from shapely.affinity import affine_transform as T
from shapely.affinity import rotate as R

def get_storhogen_data():
    url_metar = "https://api.met.no/weatherapi/tafmetar/1.0/metar.txt?icao=ENSG"

    metar_strings = (
        requests
        .get(url_metar)
        .content
        .decode()
        .split("\n")
    )
    metar_strings = [s for s in metar_strings if s !=""]
    #%%
    metar_observations = []
    obs_list = []
    for i, s in enumerate(metar_strings):
        try:
            obs = Metar.Metar(s)
            metar_observations.append(obs)
            obs_storhogen = obs.string().split(" ")[-1]
            sh_dir = int(obs_storhogen[:3])
            sh_windspeed = int(obs_storhogen[3:5])

            obs_list.append(
                {
                    'location' : obs.station_id,
                    'time' : obs.time,
                    'wind_angle' : sh_dir, 
                    'wind_strength' : sh_windspeed*0.51
                    }
            )
        except:
            pass
    df = pd.DataFrame(obs_list)
    df['lat'] = 61.1732881
    df['lon'] = 7.1195861
    df['altitude'] = 1170
    df['name'] = "Storhogen"
    df['hours_since_reading'] = (datetime.now()-df['time']).apply(lambda x: np.round(x.seconds/3600,1))
    return df


#%%
def collect_holfuy_data():
    modva_raw = json.loads(requests.get(f"http://api.holfuy.com/live/?s=1550&pw={st.secrets['holfuy_secret']}=JSON&tu=C&su=m/s").content)

    modva = {
        'stationId' : 1550,
        'lat' : 61.3454358,
        'lon' : 7.1977754,
        'altitude' : 900,
        'name' : "Modvaberget",
        'hours_since_reading' : 0
    }

    modva['lat'] = 61.3454358
    modva['lon'] = 7.1977754
    modva['altitude'] = 900
    modva['name'] = modva_raw['stationName']
    modva['wind_strength']= modva_raw['wind']['speed']
    modva['wind_angle'] = modva_raw['wind']['direction']
    modva['s'] = np.sqrt(modva_raw['wind']['speed'])/200
    return pd.DataFrame([modva])

#%% NETATMO AUTH
def collect_netatmo_data():
    # use existing package to fix authentication of netatmo
    import netatmo
    ws = netatmo.WeatherStation( {
            'client_id': st.secrets['netatmo_client_id'],
            'client_secret': st.secrets['netatmo_client_secret'],
            'username': st.secrets['netatmo_username'],
            'password': st.secrets['netatmo_password'],
            'device': st.secrets['netatmo_device']} )
    ws.get_data()
    #%% NETATMO
    # lon / lat
    coord_sw = [6.4635085,61.0616599]
    coord_ne = [6.76,61.646145]

    r = requests.get(f"https://api.netatmo.com/api/getpublicdata?lat_ne={coord_ne[1]}&lon_ne={coord_ne[0]}&lat_sw={coord_sw[1]}&lon_sw={coord_sw[0]}&required_data=wind&filter=true",  headers={'Authorization': f'Bearer {ws._access_token}'})
    msg = json.loads(r.content)
    if r.status_code!=200: 
        print(r.content)
    d = msg['body'][1]

    ## Organize data into dataframe
    data = []
    for d in msg['body']:
        obs = {}
        obs['lat'] = d['place']['location'][1]
        obs['lon'] = d['place']['location'][0]
        obs['altitude'] = d['place']['altitude']
        obs['name'] = f"netatmo-{d['place'].get('street')}"
        data.append(obs)

        for mac, observations in d['measures'].items():
            if "wind_angle" in observations.keys():
                if observations['wind_angle'] >= 0:
                    for s in ['wind_angle','wind_strength','wind_timeutc','gust_strength']:
                        obs[s] = observations[s]

    df = pd.DataFrame(data).dropna()
    df['hours_since_reading'] = np.round(df['wind_timeutc'].apply(lambda epoch: (datetime.utcnow()-datetime.utcfromtimestamp(epoch)).seconds/3600), 1)

    df['s'] = np.sqrt(df['wind_strength'])/500
    return df

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
#%%

#%% Presentation
st.title("Flyinfo Sogn")

#%% Modvoberget
st.components.v1.iframe(
    src="https://widget.holfuy.com/?station=1550&su=m/s&t=C&lang=en&mode=detailed",
    height=250
)
st.components.v1.iframe(
    src="https://widget.holfuy.com/?station=1550&su=m/s&t=C&lang=en&mode=average&avgrows=32",
    height=170
)
#%%
st.subheader("Storhogen")
try:
    
    df_storhogen = get_storhogen_data()
    last_obs = df_storhogen.tail(1).to_dict("records")[0]
    
    st.markdown(f"""
    **time:** \t {last_obs['time']}  
    **Hours since reading **: \t {last_obs['hours_since_reading']:.1f}  
    speed:  \t {last_obs['wind_strength']:.1f} m/s  
    direction: \t {last_obs['wind_angle']}°
    """)
except:
    pass

fig = px.bar_polar(
    df_storhogen.tail(1), 
    r="wind_strength", 
    theta="wind_angle", 
    color="location",
    color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.update_layout(
    dragmode=False,
    #template=None,
    showlegend=True,
    polar = dict(
      radialaxis_tickfont_size = 15,
      angularaxis = dict(
        tickfont_size=10,
        rotation=90, # start position of angular axis
        direction="clockwise"
      )
    )
)
st.plotly_chart(fig)
#%%
st.subheader("Live map")
df = collect_netatmo_data()
holfuy = collect_holfuy_data() # Append modvaberget
storhogen=df_storhogen.tail(1)
storhogen['s'] = np.sqrt(storhogen['wind_strength'])/200
df = pd.concat([holfuy, storhogen.tail(1), df ])
fig = plot_wind_arrows(df)
fig

#%% WINDY
st.subheader("Windy nå 1500moh")
st.components.v1.iframe(
    src="https://embed.windy.com/embed2.html?lat=61.010&lon=7.015&detailLat=61.249&detailLon=7.086&width=650&height=450&zoom=8&level=850h&overlay=wind&product=ecmwf&menu=&message=true&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=true&metricWind=m%2Fs&metricTemp=%C2%B0C&radarRange=-1",
    height=450
)

#%% Vindnå.no
st.components.v1.iframe(
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