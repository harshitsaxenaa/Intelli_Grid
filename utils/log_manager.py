import json
import os
from datetime import datetime

LOG_FILE = 'logs/severity_log.json'
DETECTION_STATE_FILE = 'utils/detection_state.json'

def load_json(file_path, default):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(default, f, indent=4)
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def start_new_session():
    logs = load_json(LOG_FILE, [])
    detection_state = {
        'fire': 'No',
        'smoke': 'No',
        'blood': 'No',
        
        'lying_person': 'No'
    }

    session_id = len(logs)
    new_session = {
        'session_id': session_id,
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'running',
        'detection_state': detection_state,
        'severity_score': 0,
        'lat': 28.6139,  # default New Delhi
        'lon': 77.2090,
        'location': "New Delhi, India"
    }

    logs.append(new_session)
    save_json(LOG_FILE, logs)
    save_json(DETECTION_STATE_FILE, detection_state)
    return session_id

def update_detection(session_id, label):
    logs = load_json(LOG_FILE, [])
    detection_state = load_json(DETECTION_STATE_FILE, {})

    if detection_state.get(label, 'No') == 'No':
        detection_state[label] = 'Yes'
        save_json(DETECTION_STATE_FILE, detection_state)

        session = logs[session_id]
        session['detection_state'][label] = 'Yes'
        session['severity_score'] = calculate_severity(session['detection_state'])
        logs[session_id] = session
        save_json(LOG_FILE, logs)

def update_location(session_id, lat, lon, location):
    logs = load_json(LOG_FILE, [])
    
    logs[session_id]['lat'] = lat
    logs[session_id]['lon'] = lon
    logs[session_id]['location'] = location
    save_json(LOG_FILE, logs)


def calculate_severity(state):
    weights = {
        'accident': 3,
        'fire': 3,
        'smoke': 2,
        'blood': 2,
        'lying_person': 1
    }
    score = sum(weights[key] for key, value in state.items() if value == 'Yes')
    return score

def end_session(session_id):
    logs = load_json(LOG_FILE, [])
    logs[session_id]['status'] = 'completed'
    save_json(LOG_FILE, logs)
