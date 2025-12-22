import requests
from config import API_BASE

def adjust_inventory(payload: dict):
    res = requests.post(f"{API_BASE}/transactions", json=payload)
    res.raise_for_status()
    return res.json()
