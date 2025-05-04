import streamlit as st
from streamlit_autorefresh import st_autorefresh
import json

st.set_page_config(page_title="Ambulance Dashboard", layout="centered")

# Password protection
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("password_form"):
            password = st.text_input("Enter Password", type="password")
            submitted = st.form_submit_button("Unlock")
            if submitted:
                if password == "1234abc":
                    st.session_state.authenticated = True
                else:
                    st.error("Incorrect password")
                    st.stop()
            else:
                st.stop()

check_password()

ACCIDENT_DB = 'logs/severity_log.json'
ROUTE_DB = 'logs/routes_db.json'

# Auto-refresh 
st_autorefresh(interval=4000, key="ambulance_refresh")

st.title(" Ambulance Alert for Nearby Accidents")

try:
    with open(ACCIDENT_DB, 'r') as f:
        data = json.load(f)
        if not data:
            st.warning("No recent accident data available.")
            st.stop()
        latest = data[-1]
except:
    st.error("Error loading accident data.")
    st.stop()

# Display
start_time = latest.get('start_time', 'N/A')
severity = latest.get('severity_score', 0)
status = latest.get('status', '').lower()
location = "ABC"  # Placeholder

st.subheader("⚠️ Accident Detected Nearby")
st.write(f"**Date & Time:** {start_time}")
st.write(f"**Location:** {location}")
st.write(f"**Severity Score:** {severity}")

# Show Accept button if not already accepted
if status != "ambulance enroute":
    if st.button("🚑 Accept Ambulance Request"):
        try:
            latest['status'] = 'ambulance enroute'
            data[-1] = latest
            with open(ACCIDENT_DB, 'w') as f:
                json.dump(data, f, indent=4)
            st.success("🚑 Request Accepted. Ambulance is now enroute!")
            st.experimental_rerun()
        except:
            st.error("")
else:
    st.success("✅ Ambulance has already accepted the request and is enroute.")

    
    try:
        with open(ROUTE_DB) as f:
            routes = json.load(f)

        relevant_routes = [r for r in routes if r['location'] == location]

        if relevant_routes:
            best_route = min(relevant_routes, key=lambda r: r['estimated_time_min'])
            st.markdown("### 🚑 Best Route to Accident")
            st.write(f"**Route ID:** {best_route['route_id']}")
            st.write(f"**Distance:** {best_route['distance_km']} km")
            st.write(f"**Estimated Time:** {best_route['estimated_time_min']:.2f} mins")
        else:
            st.warning("No route info available for the location.")
    except:
        st.error("Error loading route data.")

    # Reset button 
    st.markdown("---")
    if st.button("🔁 Reset Request"):
        try:
            latest['status'] = 'pending'
            data[-1] = latest
            with open(ACCIDENT_DB, 'w') as f:
                json.dump(data, f, indent=4)
            st.success("Request reset to pending.")
            st.experimental_rerun()
        except:
            st.error("")
