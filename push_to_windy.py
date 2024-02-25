import utils
import requests

#%% get ecowitt data

""" example data format
# print(data['Barten']:
{'lat': 61.2477458,
 'lon': 7.0883309,
 'altitude': 700,
 'name': 'Barten',
 'measurements':     temperature  wind_strength  wind_gust  wind_angle                time
 0           1.0            0.4        1.8        15.0 2024-02-25 09:00:00
 1           1.0            0.4        1.8        16.0 2024-02-25 08:30:00
 2           0.5            0.4        1.8        14.0 2024-02-25 08:00:00
 3           0.7            0.0        0.0        14.0 2024-02-25 07:30:00

 # print(data['Storhogen']):
 {'lat': 61.1732881,
 'lon': 7.1195861,
 'altitude': 1170,
 'name': 'Storhogen',
 'measurements':                   time  wind_angle  wind_strength
 0  2024-02-25 08:50:00         200           7.14
 1  2024-02-25 08:20:00         200           9.18
 2  2024-02-25 07:50:00         200           9.18
 """

def push_to_windy(station, data):
    """
    station = "Barten"
    data = all_data[station]

    # info and weather parameters:
    Info Part Parameters
        Station record in the database is created as soon as required info params are uploaded (station, lat, lon).

        station - 32 bit integer; required for multiple stations; default value 0; alternative names: si, stationId
        shareOption - text one of: Open, Only Windy, Private; default value is Open
        name - text; user selected station name
        latitude - number [degrees]; required; north–south position on the Earth`s surface
        longitude - number [degrees]; required; east–west position on the Earth`s surface
        elevation - number [metres]; height above the Earth's sea level (reference geoid); alternative names: elev, elev_m, altitude
        tempheight - number [metres]; temperature sensor height above the surface; alternative names: agl_temp
        windheight - number [metres]; wind sensors height above the surface; alternative names: agl_wind
    Measurements
        station - 32 bit integer; required for multiple stations; default value 0; alternative names: si, stationId
        time - text; iso string formated time "2011-10-05T14:48:00.000Z"; when time (or alternative) is NOT present server time is used
        dateutc - text; UTC time formated as "2001-01-01 10:32:35"; (alternative to time)
        ts - unix timestamp [s] or [ms]; (alternative to time)
        temp - real number [°C]; air temperature
        tempf - real number [°F]; air temperature (alternative to temp)
        wind - real number [m/s]; wind speed
        windspeedmph - real number [mph]; wind speed (alternative to wind)
        winddir - integer number [deg]; instantaneous wind direction
        gust - real number [m/s]; current wind gust
        windgustmph - real number [mph]; current wind gust (alternative to gust)
        rh - real number [%]; relative humidity ; alternative name: humidity
        dewpoint - real number [°C];
        pressure - real number [Pa]; atmospheric pressure
        mbar - real number [milibar, hPa]; atmospheric pressure alternative
        baromin - real number [inches Hg]; atmospheric pressure alternative
        precip - real number [mm]; precipitation over the past hour
        rainin - real number [in]; rain inches over the past hour (alternative to precip)
        uv - number [index];
        
        #Example get request
        https://stations.windy.com/pws/update/XXX-API-KEY-XXX?winddir=230&windspeed=12&windgust=12&temp=10&rain=0&baromin=29.1&dewpoint=5&humidity=90
    """
    station_info = stations.get(station)
    if not station_info:
        print(f"No info found for station {station}")
        return

    station_token = windy_token
    station_id = station_info['stationId']

    # Get the latest observation
    latest_observation = data['measurements'].iloc[0]

    # Prepare the parameters for the GET request
    params = {
        'station': station_id,
        'lat': data['lat'],
        'lon': data['lon'],
        'time' : latest_observation['time'].strftime('%Y-%m-%d %H:%M:%S'),
        'winddir': latest_observation.get('wind_angle'),
        'wind': latest_observation.get('wind_strength'),
        'gust': latest_observation.get('wind_gust'),
        'temp': latest_observation.get('temperature'),  # Some stations might not have all measurements
        'rain': latest_observation.get('rain'),
        'baromin': latest_observation.get('baromin'),
        'dewpoint': latest_observation.get('dewpoint'),
        'humidity': latest_observation.get('humidity'),
    }

    # Remove any parameters that weren't in the data
    params = {k: v for k, v in params.items() if v is not None}

    # Send the GET request to Windy
    response = requests.get(f"https://stations.windy.com/pws/update/{station_token}", params=params)

    # Check the response
    if response.status_code == 200:
        print(f"Data successfully sent to Windy for station {station}. values: {params}")
        logging.info(f"Data successfully sent to Windy for station {station}. values: {params}")
    else:
        print(f"Failed to send data to Windy for station {station}. Response code: {response.status_code}")
        logging.error(f"Failed to send data to Windy for station {station}. Response code: {response.status_code}")
# Push latest barten observation to windy with get request

# %%
import logging
if __name__ == "__main__":
    windy_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjaSI6MjA3NTQ3MCwiaWF0IjoxNzA4ODUxMjU1fQ.Ls4TTT8MI4ZyGlX2vRuSiXDLCtGFoy4qOXvSfFTnAco"
    stations = {
        'Barten' : {
            'stationId' : "0",
        },
        'Storhogen' : {
            'stationId' : "1",
        },
    }
    all_data = utils.get_weather_measurements(lookback=12)

    for station in stations.keys():
        push_to_windy(station, all_data[station])