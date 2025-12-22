import requests
from config import API_BASE

def list_transactions(params=None):
    response = requests.get(
        f"{API_BASE}/transactions",
        params=params
    )

    if response.status_code != 200:
        print("STATUS:", response.status_code)
        print("BODY:", response.text)

    response.raise_for_status()
    return response.json()

def create_transaction(payload: dict):
    response = requests.post(
        f"{API_BASE}/transactions",
        json=payload
    )
    response.raise_for_status()
    return response.json()