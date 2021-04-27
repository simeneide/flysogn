#%% 
import streamlit as st
import requests
import pandas as pd
from metar import Metar
import plotly.graph_objects as go

url_metar = "https://api.met.no/weatherapi/tafmetar/1.0/metar.txt?icao=ENSG"
#url_metar = "https://www.ogimet.com/display_metars2.php?lang=en&lugar=ENSG&tipo=ALL&ord=REV&nil=SI&fmt=txt&ano=2021&mes=04&day=01&hora=09&anof=2021&mesf=04&dayf=27&horaf=09&minf=59&send=send"

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
                'wind_dir' : sh_dir, #obs.wind_dir.value(),
                'wind_speed' : sh_windspeed*0.51# obs.wind_speed.value()
                }
        )
    except:
        pass

df = pd.DataFrame(obs_list)
#%% Presentation
st.title("Flyinfo Storhogen")
last_obs = obs_list[-1]
fig = go.Figure(go.Barpolar(
    r=[last_obs['wind_speed']],
    theta=[last_obs['wind_dir']],
    width=[20],
    #marker_color=["#E4FF87"],
    marker_line_color="black",
    marker_line_width=2,
    opacity=0.8
))

fig.update_layout(
    template=None,
    polar = dict(
      radialaxis_tickfont_size = 8,
      angularaxis = dict(
        tickfont_size=10,
        rotation=90, # start position of angular axis
        direction="clockwise"
      )
    )
)
st.plotly_chart(fig)
#%% Historical data
fig = go.Figure(data=go.Scatter(x=df.time, y=df.wind_dir))
fig.update_layout(title="Vindretning storhogen", xaxis_title="Tid", yaxis_title="Vindretning [°]")
st.plotly_chart(fig)

fig = go.Figure(data=go.Scatter(x=df.time, y=df.wind_speed))
fig.update_layout(title="Vindstyrke storhogen", xaxis_title="Tid", yaxis_title="Vind [m/s]")
st.plotly_chart(fig)

#%% WINDY
st.subheader("Windy nå 1500moh")
st.components.v1.html("""
<iframe width="650" height="450" src="https://embed.windy.com/embed2.html?lat=61.010&lon=7.015&detailLat=61.249&detailLon=7.086&width=650&height=450&zoom=8&level=850h&overlay=wind&product=ecmwf&menu=&message=true&marker=&calendar=now&pressure=&type=map&location=coordinates&detail=true&metricWind=m%2Fs&metricTemp=%C2%B0C&radarRange=-1" frameborder="0"></iframe>
""", height=450)

st.text("""
Visualisering av metar-data fra storhogen for lettere å få en oversikt over vindforhold akkurat nå.
Repo: https://github.com/simeneide/flysogn
Data er hentet fra api.met.no
""")