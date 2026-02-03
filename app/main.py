from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=["*"],   # MVP only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Startup event
# ----------------------------
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

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
app.include_router(products_router.router, prefix="/api/v1", tags=["Products"])
app.include_router(rfid_router.router, prefix="/api/v1", tags=["RFID"])
app.include_router(transactions_router.router, prefix="/api/v1", tags=["Transactions"])
app.include_router(history_router.router, prefix="/api/v1", tags=["History"])
app.include_router(analytics_router.router, prefix="/api/v1", tags=["Analytics"])