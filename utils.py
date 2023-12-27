#%%
import requests
import streamlit as st
from metar import Metar
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
#%%
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
                    'wind_strength' : sh_windspeed*0.51 # kt to m/s
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

import pandas as pd
from ecowitt_net_get import ecowitt_get_history, ecowitt_get_realtime

def get_historical_ecowitt(lookback=1):
    variables = ['outdoor.temperature', 'wind.wind_speed', 'wind.wind_gust', 'wind.wind_direction']
    selection = ",".join(variables)
    start_date = (datetime.now() - timedelta(days=lookback)).replace(minute=0, second=0, microsecond=0)
    end_date = datetime.now()

    data = ecowitt_get_history(start_date, end_date, call_back=selection, cycle_type='5min')
    # transform into arrays
    
    dfs= {}
    for key, value in data.items():
        if isinstance(value, dict):
            dfs[key] = pd.DataFrame.from_dict(value['list'], orient='index')
    # join into one df with columns
    df = pd.concat(dfs, axis=1)
    # Remove multi index
    df.columns = df.columns.droplevel(1)
    # only keep last name after last . in column name:
    df.columns = df.columns.str.split('.').str[-1]
    df.rename(columns={'wind_speed':'wind_strength', 'wind_direction' : 'wind_angle'}, inplace=True)
    df['time'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['lat'] = 61.2477458
    df['lon'] = 7.0883309
    df['name'] = "Barten"
    df['altitude'] = 700
    df['s'] = np.sqrt(df['wind_strength'])/200
    #df['time'] = pd.to_datetime(df['time'])
    df['hours_since_reading'] = df['time'].apply(lambda x: np.round((datetime.now()-x).total_seconds()/3600,1))
    return df


import json
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
    # NETATMO
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
# %%
