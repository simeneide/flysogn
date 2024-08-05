#!/usr/bin/python
import json
import requests
import json
import pandas as pd
import streamlit as st
# python nve_utils.py -a "8RK0Dwiz5U+x2t5WhMFHmg==" -s 12.209.0 -p 1000 -r 60 -t "2018-01-01T10:00:00/2018-01-14T20:00:00"
# get parameters
# response = requests.get("https://hydapi.nve.no/api/v1/Parameters", headers=request_headers)    
# parameter_list = json.loads(response.content)['data']
# for parameter in parameter_list:
#     print(parameter)
# {'parameter': 14, 'parameterName': 'Vindretning', 'parameterNameEng': 'Wind direction', 'unit': '°'}
# {'parameter': 15, 'parameterName': 'Vindhastighet', 'parameterNameEng': 'Wind speed', 'unit': 'm/s'}
# {'parameter': 17, 'parameterName': 'Lufttemperatur', 'parameterNameEng': 'Air temperature', 'unit': '°C'}
from datetime import datetime, timedelta
def get_reference_time(days=7):
    # Get the current date and time
    now = datetime.now()
    
    start_of_week = now - timedelta(days=days)
    
    # Format the start and end dates in the required format
    start_str = start_of_week.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Combine the start and end dates into the desired format
    latest_week = f"{start_str}/{end_str}"
    return latest_week

def fetch_flatbre_data(days=7):
    station="78.19.0"
    parameter="14,15"
    resolution_time="0"
    #reference_time="2018-01-01T10:00:00/2018-01-14T20:00:00"


    # Example usage
    reference_time = get_reference_time()
    print(reference_time)
    api_key = st.secrets['nve_api_key']

    baseurl = "https://hydapi.nve.no/api/v1/Observations?StationId={station}&Parameter={parameter}&ResolutionTime={resolution_time}"

    url = baseurl.format(station=station, parameter=parameter,
                            resolution_time=resolution_time)

    if reference_time is not None:
        url = "{url}&ReferenceTime={reference_time}".format(
            url=url, reference_time=reference_time)

    print(url)

    request_headers = {
        "Accept": "application/json",
        "X-API-Key": api_key
    }

    response = requests.get(url, headers=request_headers)
    response.status_code
    json_data = json.loads(response.content)['data']


    dfs = {}
    # Loop through each JSON object
    for entry in json_data:
        #if len(entry['observations']):
        #    Warning(f"Empty observations for {entry}")
        #    continue
        parameter_name = entry['parameterNameEng']
        observations = entry['observations']
        
        # Create a DataFrame for the observations
        df = pd.DataFrame(observations)
        
        # Rename the 'value' column to the parameter name
        df = df.rename(columns={'value': parameter_name})
        # drop columns except time and parameter_name
        df = df[['time', parameter_name]]
        # Store the DataFrame in the dictionary
        dfs[parameter_name] = df

    # Merge all DataFrames on the 'time' column
    final_df = None
    for df in dfs.values():
        if final_df is None:
            final_df = df
        else:
            final_df = pd.merge(final_df, df, on='time', how='outer')


    final_df.rename(columns={"Wind direction" : "wind_angle", "Wind speed": "wind_strength"}, inplace=True)
    # Print the final DataFrame
    return final_df

def get_flatbre_data(days=7):
    measurements = fetch_flatbre_data(days)
    out = {'lat': 61.47384, 
            'lon': 6.79536,
            'altitude': 989,
            'name': 'Storhogen',
            'measurements': measurements}
    return out
if __name__ == "__main__":
    get_flatbre_data()