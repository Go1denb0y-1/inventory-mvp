from fastapi import FastAPI

from app.database import engine
from app.models import Base

from app.routers import (
    products_router,
    rfid_router,
    transactions_router,
    history_router,
    analytics_router
)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Inventory Management System")

# Register routers
app.include_router(products_router.router, prefix="/api/v1", tags=["Products"])
app.include_router(rfid_router.router, prefix="/api/v1", tags=["RFID"])
app.include_router(transactions_router.router, prefix="/api/v1", tags=["Transactions"])
app.include_router(history_router.router, prefix="/api/v1", tags=["History"])
app.include_router(analytics_router.router, prefix="/api/v1", tags=["Analytics"])

<<<<<<< HEAD
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
=======
>>>>>>> 2e52078ef15e6ab30b08cd173882eef789367aaf
