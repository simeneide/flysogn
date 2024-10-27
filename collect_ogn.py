import streamlit as st
from ogn.client import AprsClient
from ogn.parser import parse, ParseError
from threading import Thread
import time  # To include small delays for improved responsiveness

lon_min = 6
lat_min = 60#61
lon_max = 9 #7.5
lat_max = 62

# Known devices
known_devices = {
    'OGNFD65AF' : 'Simen'
}

# Latest position of a specific aircraft
if not hasattr(st, 'latest_pos'):
    st.latest_pos = {}

def process_beacon(raw_message):
    try:
        beacon = parse(raw_message)
        if beacon['aprs_type'] == "position":
            if (lon_min < beacon['longitude'] < lon_max) and (lat_min < beacon['latitude'] < lat_max):
                print(beacon)
                beacon_keep_vars = ['timestamp','name', 'latitude', 'longitude','altitude','beacon_type']
                beacon = {k: v for k, v in beacon.items() if k in beacon_keep_vars}
                name = known_devices.get(beacon['name'], beacon['name'])
                st.latest_pos[name] = beacon
                #print(beacon)
    except (ParseError, NotImplementedError):
        pass
    except Exception as e:
        print(e)

def run_ogn_client():
    client = AprsClient(aprs_user='N0CALL')
    client.connect()
    try:
        client.run(callback=process_beacon, autoreconnect=True)
    except KeyboardInterrupt:
        client.disconnect()

st.cache_resource()
def start_client():
    thread = Thread(target=run_ogn_client)
    thread.daemon = True
    thread.start()
    return thread

if __name__ == "__main__":

    st.title("Aircraft Tracker")
    if not hasattr(st, 'client_started'):
        start_client()
        st.client_started = True
    
    # Continuously display data
    st.subheader("Latest Positions")
    if st.latest_pos:
        for name, pos in st.latest_pos.items():
            st.write(f"Received {name}: {pos['timestamp'],pos['latitude']}, {pos['longitude']}")

    # Update the Streamlit display every few seconds
    time.sleep(2)  # Sleep for a short duration before rerunning
    st.rerun()
