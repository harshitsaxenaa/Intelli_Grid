import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import requests
import json

st.set_page_config(page_title="Nearby Hospitals & Alerts", layout="wide")

st.title("Casualty Alert & Nearby Hospitals")

try:
    with open("logs/severity_log.json", "r") as f:
        data = json.load(f)
        latest = data[-1]
        location = latest.get("location", "Unknown Location")
        lat = float(latest.get("lat", 28.6139))
        lon = float(latest.get("lon", 77.2090))

except:
    st.error("Unable to load casualty data.")
    st.stop()

st.subheader(f"⚠️ Casualty Zone: {location}")

# map with red alert marker
m = folium.Map(location=[lat, lon], zoom_start=14)
folium.Marker(
    location=[lat, lon],
    popup="🚨 Recent casualty Reported Here",
    icon=folium.Icon(color="red", icon="warning-sign")
).add_to(m)

with st.spinner("Fetching nearby hospitals..."):
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q=hospital&limit=10&bounded=1&viewbox={lon-0.02},{lat+0.02},{lon+0.02},{lat-0.02}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        results = response.json()

        if results:
            st.markdown("### Nearby Hospitals")
            for place in results:
                name = place.get("display_name", "Hospital")
                lat_h = float(place["lat"])
                lon_h = float(place["lon"])

                folium.Marker(
                    location=[lat_h, lon_h],
                    popup=name,
                    icon=folium.Icon(color="blue", icon="plus-sign")
                ).add_to(m)
                st.write(f"🔹 {name}")
        else:
            st.warning("No hospitals found nearby.")
    except Exception as e:
        st.error(f"Error fetching hospitals: {e}")

# Display map
st_folium(m, width=1000, height=500)
