from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InventoryHistory
from app.schemas import InventoryHistoryOut

router = APIRouter(
    prefix="/history",
    tags=["Inventory History"]
)

@router.get("/", response_model=list[InventoryHistoryOut])
def get_history(
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    db: Session = Depends(get_db)
):
    """
    Retrieve inventory history with optional date filtering.
    """

    query = db.query(InventoryHistory)

    if start_date:
        query = query.filter(InventoryHistory.timestamp >= start_date)

    if end_date:
        query = query.filter(InventoryHistory.timestamp <= end_date)

    return query.order_by(InventoryHistory.timestamp.desc()).all()
