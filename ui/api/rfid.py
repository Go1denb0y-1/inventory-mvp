import requests
from config import API_BASE

def scan_rfid(payload: dict):
    res = requests.post(f"{API_BASE}/rfid/scan", json=payload)
    res.raise_for_status()
    return res.json()
