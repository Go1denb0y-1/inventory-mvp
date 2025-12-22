import requests
from config import API_BASE

def list_products():
    res = requests.get(f"{API_BASE}/products")
    res.raise_for_status()
    return res.json()

def create_product(payload: dict):
    res = requests.post(f"{API_BASE}/products", json=payload)
    res.raise_for_status()
    return res.json()

def update_product(sku: str, payload: dict):
    res = requests.patch(f"{API_BASE}/products/{sku}", json=payload)
    res.raise_for_status()
    return res.json()