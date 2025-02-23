#%%
import requests
import pandas as pd
import streamlit as st
# Add secrets to env
st.secrets
from datetime import datetime, timedelta
import os
#%% Get sources
def fetch_metno_station(lookback=7):
    """
    lookback=7 # days
    """
    endpoint = 'https://frost.met.no/sources/v0.jsonld'
    r = requests.get(endpoint, auth=(os.environ['METNO_CLIENT_ID'],''), headers={'Accept': 'application/json'})
    sources = pd.DataFrame(r.json()['data'])

    ## 
    # Expanding the 'geometry' column
    expanded_geometry = sources['geometry'].apply(pd.Series)

    # If you specifically want the coordinates
    coordinates = expanded_geometry['coordinates'].apply(pd.Series)
    coordinates.columns = ['longitude', 'latitude']

    # Concatenate the new DataFrame with the original DataFrame
    sources = pd.concat([sources, coordinates], axis=1)

    #%% Filter sources
    use_ids = ['SN55000','SN55709']
    met_no_sources = sources[sources['id'].isin(use_ids)].to_dict(orient='records')

    sources_use = sources[sources['municipality'].isin(["SOGNDAL", "LUSTER","VIK"])]
    stations_df = sources_use[['id','shortName','masl','longitude','latitude']]
    #%%
    # Get temperature data for stations

    current_date = datetime.now()
    start_date = current_date - timedelta(days=lookback)
    end_date = current_date + timedelta(days=1)
    referencetime = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"



    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    # elementIds can be found at https://frost.met.no/elementtable
    parameters = {
        'sources': ",".join(stations_df['id'].values),
        'elements': 'air_temperature,wind_speed,wind_from_direction,max(wind_speed_of_gust PT10M)', # , wind_speed, wind_from_direction
        'referencetime': referencetime,
    }

    r = requests.get(endpoint, parameters, auth=(os.environ['METNO_CLIENT_ID'],''))
    # Extract JSON data
    json = r.json()
    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
    else:
        print('Error! Could not get data from frost.met.no. Response code:', r.status_code)
        return {}
    # %%
    # This will return a Dataframe with all of the observations in a table format
    dfs = []
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        dfs.append(row)

    df = pd.concat(dfs).reset_index(drop=True)
    df['time'] = pd.to_datetime(df['referenceTime'])

    df = df[['time', 'sourceId','elementId', 'value']]

    # rename elementId names:
    df.loc[df['elementId'] == 'air_temperature', 'elementId'] = 'temperature'
    df.loc[df['elementId'] == 'wind_from_direction', 'elementId'] = 'wind_direction'
    df.loc[df['elementId'] == 'max(wind_speed_of_gust PT10M)', 'elementId'] = 'wind_gust'

    df['sourceId'] = df['sourceId'].apply(lambda x: x.split(':')[0])


    stations = {}
    station_list = stations_df.to_dict(orient='records')

    # station_dict = station_list[1]
    for station_dict in station_list:
        # Filter the DataFrame for the current station
        station_data = df[df['sourceId'] == station_dict['id']]

        station_data_agg = station_data.groupby(['time', 'elementId','sourceId']).mean().reset_index()

        if station_data_agg.empty:
            continue

        # Pivot the table to have `time` as index and `elementId` as columns
        station_pivot = station_data_agg.pivot(index='time', columns='elementId', values='value').reset_index()
        
        station_dict = {
            'measurements' : station_pivot,
            'lat' : station_dict['latitude'],
            'lon' : station_dict['longitude'],
            'altitude' : station_dict['masl'],
            'name' : station_dict['shortName'],
            'station' : station_dict['id']}
        # Store the DataFrame in the dictionary
        stations[station_dict['name']] = station_dict
    return stations
# Now `station_dfs` contains a DataFrame for each station with columns for each element
# %%

if __name__ == "__main__":
    stations = fetch_metno_station()
    for key, val in stations.items():
        print(key)
        print(val['measurements'].head())
# %%
