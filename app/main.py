from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager
from app.database import engine
from app.models import Base

from app.routers import (
    products_router,
    rfid_router,
    transactions_router,
    history_router,
    analytics_router,
    
)

app = FastAPI(title="Inventory Management System")

# ----------------------------
# CORS (allow Streamlit UI)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Startup event
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP: Runs when the app starts ---
    # 1. Create Tables
    Base.metadata.create_all(bind=engine)
    
    # 2. DEBUG: Log every registered route to the console
    print("\n--- REGISTERED ROUTES ---")
    for route in app.routes:
        methods = ", ".join(route.methods) if hasattr(route, 'methods') else "N/A"
        print(f"PATH: {route.path:40} | METHODS: {methods}")
    print("-------------------------\n")
    
    yield
    # --- SHUTDOWN: Runs when the app stops ---
    pass

# ----------------------------
# Health & root endpoints
# ----------------------------
@app.get("/")
def root():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/db-test")
def db_test():
    with engine.connect() as conn:
        return {"db": "connected"}

# ----------------------------
# Routers
# ----------------------------
# Temporary change in main.py
app.include_router(products_router.router, prefix="/api/v1/test-products", tags=["Products"])
# app.include_router(rfid_router.router, prefix="/api/v1", tags=["RFID"])
app.include_router(transactions_router.router, prefix="/api/v1", tags=["Transactions"])
# app.include_router(history_router.router, prefix="/api/v1", tags=["History"])
# app.include_router(analytics_router.router, prefix="/api/v1", tags=["Analytics"])

