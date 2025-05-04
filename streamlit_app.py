import streamlit as st
import json
import os

st.set_page_config(page_title="Incident Monitor Dashboard", layout="wide")

LOG_FILE = 'logs/severity_log.json'
ACCIDENT_DB = 'logs/accident_db.json'

# Load logs
def load_logs():
    try:
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# Save logs
def save_logs(logs):
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

# Delete session
def delete_session(session_id):
    logs = load_logs()
    logs = [session for session in logs if session.get('session_id') != session_id]
    save_logs(logs)
    st.success(f"Session {session_id} deleted successfully.")
    st.rerun()

# Load 
def load_accident_db():
    try:
        with open(ACCIDENT_DB, 'r') as dbf:
            return json.load(dbf)
    except:
        return {}

# Streamlit 
st.title("🚨 Emergency Incident Monitoring")

logs = load_logs()
accident_data = load_accident_db()
accident_time = accident_data.get("datetime", "")
if not logs:
    st.warning("No sessions available.")
else:
    for session in reversed(logs):
        session_id = session.get("session_id", "Unknown ID")
        session_time = session.get('start_time', '')

        with st.container():
            st.subheader(f"Session ID: {session_id} - Status: {session.get('status', 'N/A').capitalize()}")
            st.write(f"Start Time: {session_time}")

            # Match and show ambulance status
            if accident_time and accident_time == session_time:
                if accident_data.get("ambulance_enroute"):
                    st.info("🚑 Ambulance Enroute")
                else:
                    st.warning("⏳ Ambulance not yet dispatched.")
            blood_detected = session.get('detection_state', {}).get('blood_detected', 'No') == 'Yes'
            severity_score = session.get('severity_score', 0)

            if blood_detected or severity_score > 5:
                st.error("🏥 Hospital Alerted")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Detection States")
                for key, value in session.get('detection_state', {}).items():
                    emoji = "✅" if value == 'Yes' else "❌"
                    st.write(f"- **{key.replace('_', ' ').title()}**: {emoji} {value}")

            with col2:
                st.markdown("### Severity Score")
                score = session.get('severity_score', 0)
                st.metric(label="Score", value=score)
                if score >= 6:
                    st.error("🚨 Critical Severity")
                elif score >= 3:
                    st.warning("⚠️ Moderate Severity")
                elif score > 0:
                    st.info("ℹ️ Low Severity")
                else:
                    st.success("✅ No Threat Detected")

            #Delete session 
            if st.button(f"Delete Session {session_id}", key=f"del_{session_id}"):
                delete_session(session_id)
