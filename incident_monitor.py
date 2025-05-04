import json
from datetime import datetime

STATUS_FILE = "utils/detection_state.json"

def initialize_status():
    status = {
        "accident": False,
        "fire": False,
        "smoke": False,
        "blood": False,
        "fall": False,
        "last_updated": str(datetime.now()),
        "severity_score": 0
    }
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=4)

def update_status(detected_labels):
    score = 0
    weights = {
        "accident": 3,
        "fire": 5,
        "smoke": 2,
        "blood": 4,
        "fall": 3
    }

    status = {label: False for label in weights}
    for label in detected_labels:
        status[label] = True
        score += weights[label]

    status["last_updated"] = str(datetime.now())
    status["severity_score"] = score

    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=4)

def get_status():
    with open(STATUS_FILE, 'r') as f:
        return json.load(f)
