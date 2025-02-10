#%%
import requests
import streamlit as st
from metar import Metar
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import json
import nve_utils
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
                    'time' : obs.time,
                    'wind_direction' : sh_dir, 
                    'wind_speed' : sh_windspeed*0.51 # kt to m/s
                    }
            )
        except:
            pass
    df = pd.DataFrame(obs_list)
    df['time'] =  pd.to_datetime(df['time'])
    df['wind_speed'] = df['wind_speed'].round(1)
    output = {
        'lat' : 61.1732881,
        'lon' : 7.1195861,
        'altitude' : 1170,
        'name' : "Storhogen",
        'measurements' : df,
    }
    return output

import pandas as pd
from ecowitt_net_get import ecowitt_get_history, ecowitt_get_realtime

# Output of each function is a dataframe with columns:
# datetime object
# wind_speed: m/s
# wind_direction: degrees
# wind_gust: m/s
# temperatuere: C (optional)
# battery: % (optional)

def get_wundermap_data(station_id, lookback=24):
    """
    lookback=24 # hrs
    """
    url = f"https://api.weather.com/v2/pws/observations/all/1day?stationId={station_id}&format=json&units=m&apiKey={st.secrets['wundermap_api_key']}"
    data = json.loads(requests.get(url).content)
    # filter observations for the requested lookback period (in hours)
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=lookback)
    
    recs = []
    for obs in data.get("observations", []):
        # Convert observation time to a datetime object.
        # The sample uses 'obsTimeUtc' in ISO-8601 format.
        obs_time = pd.to_datetime(obs["obsTimeUtc"])
        if obs_time < start_time:
            continue
        
        # Build a record from fields in the observation.
        record = {
            "time": obs_time,
            "wind_speed": obs["metric"].get("windspeedAvg"),
            "wind_direction": obs.get("winddirAvg"),  # sometimes provided at top level
            "wind_gust": obs["metric"].get("windgustAvg"),
            "temperature": obs["metric"].get("tempAvg"),
            "pressure": obs["metric"].get("pressureMax")  # you can choose a different pressure stat if needed
        }
        recs.append(record)
    
    # Convert list of records to a Pandas DataFrame
    df = pd.DataFrame(recs).sort_values("time", ascending=False).reset_index(drop=True)
    
    # Extract station metadata using first (most recent) observation.
    if data.get("observations"):
        first_obs = data["observations"][0]
        station_lat = first_obs.get("lat")
        station_lon = first_obs.get("lon")
        station_name = first_obs.get("stationID", "Wundermap Station")
    else:
        station_lat = station_lon = station_name = None

    # Construct the output dictionary in the same format as the other data files.
    station_data = {
        "lat": station_lat,
        "lon": station_lon,
        "altitude": None,  # Not provided by wundermap data (adjust if available)
        "name": station_name,
        "measurements": df
    }
    return station_data


def get_historical_ecowitt(lookback=24):
    variables = ['outdoor.temperature', 'wind.wind_speed', 'wind.wind_gust', 'wind.wind_direction']
    selection = ",".join(variables)
    start_date = (datetime.now() - timedelta(hours=lookback))#.replace(minute=0, second=0, microsecond=0)
    end_date = datetime.now()

    data = ecowitt_get_history(start_date, end_date, call_back=selection, cycle_type='5min')
    # transform into arrays
    assert data['data.wind.wind_gust']['unit'] == "m/s", "unit of wind gust is not m/s"
    assert data['data.wind.wind_speed']['unit'] == "m/s", "unit of wind speed is not m/s"
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
    df.rename(columns={'wind_speed':'wind_speed', 'wind_direction' : 'wind_direction'}, inplace=True)
    df['time'] = pd.to_datetime(df.index)
    df.reset_index(drop=True, inplace=True)

    station = {
        'lat' : 61.2477458,
        'lon' : 7.0883309,
        'altitude' : 700,
        'name' : "Barten",
        'measurements' : df,
    }
    return station

def collect_holfuy_data(station):
    #modva_raw = json.loads(requests.get(f"http://api.holfuy.com/live/?s={station['stationId']}&pw={st.secrets['holfuy_secret']}=JSON&tu=C&su=m/s").content)
    modva_hist = json.loads(requests.get(f"http://api.holfuy.com/archive/?s={station['stationId']}&pw={st.secrets['holfuy_secret']}=JSON&tu=C&su=m/s&cnt=100&batt=true&type=1").content)
    df = pd.DataFrame(modva_hist['measurements'])
    #    'wind': {'speed': 12.8, 'gust': 14.7, 'min': 10.6, 'direction': 199},
    # Expand the wind column:
    df['wind_speed'] = df['wind'].apply(lambda x: x['speed'])
    df['wind_direction'] = df['wind'].apply(lambda x: x['direction'])
    df['wind_gust'] = df['wind'].apply(lambda x: x['gust'])
    df['time'] = pd.to_datetime(df['dateTime'])
    
    
    df = df[['time', 'wind_speed', 'wind_direction', 'wind_gust','battery','temperature']]


    
    for key, val in station.items():
        df[key] = val
    station['measurements']  = df

    return station

from utils_metno import fetch_metno_station

@st.cache_data(ttl=180)
def get_weather_measurements(lookback=24):
    """
    lookback=24 # hrs
    """
    output = {}
    ### Storhogen
    output['Storhogen'] = get_storhogen_data()

    # wundermap
    stations = ["ISOGND2", "ISOGND1","ISOLVO1", "IHYHEI1", "ILUSTE5"]
    for station_id in stations:
        try:
            output[station_id] = get_wundermap_data(station_id, lookback=lookback)
        except:
            print(f"Could not get wundermap data for station {station_id}")
            pass
    ## METNO
    metno_stations = fetch_metno_station(lookback=int(lookback/24))
    for key, val in metno_stations.items():
        output[key] = val

    # Anestølen and flatbreen
    try:
        nve_stations = nve_utils.get_flatbre_data(days=lookback/24)
        for key, val in nve_stations.items():
            output[key] = val
    except Exception as e:
        print(f"Error fetching nve data. error: {e}")
    ### Ecowitt
    try:
        output['Barten'] = get_historical_ecowitt(lookback=lookback)
    except:
        print("Could not get ecowitt data")
        pass
    ### HOLFUY
    stations = [
        {
            'stationId' : 1550,
            'lat' : 61.3454358,
            'lon' : 7.1977754,
            'altitude' : 900,
            'name' : "Modvaberget",
        },
        {
            'stationId' : 1703,
            'lat' : 61.29223,
            'lon' : 7.0328,
            'altitude' : 1105,
            'name' : "Tylderingen",
        },
        {
            'stationId' : 586,
            'lat' : 61.8849707,
            'lon' : 6.8331465,
            'altitude' : 1011,
            'name' : "Loen Skylift",
        },
        ]
    
    for station in stations:
        try:
            output[station['name']] = collect_holfuy_data(station)
        except:
            # print what is wrong
            pass

    for key, val in output.items():
        val['measurements'] = val['measurements'].sort_values('time', ascending=False).reset_index(drop=True)
    return output


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
            if "wind_direction" in observations.keys():
                if observations['wind_direction'] >= 0:
                    for s in ['wind_direction','wind_speed','wind_timeutc','gust_strength']:
                        obs[s] = observations[s]

    df = pd.DataFrame(data).dropna()
    df['hours_since_reading'] = np.round(df['wind_timeutc'].apply(lambda epoch: (datetime.utcnow()-datetime.utcfromtimestamp(epoch)).seconds/3600), 1)

    df['s'] = np.sqrt(df['wind_speed'])/500
    return df
# %%

