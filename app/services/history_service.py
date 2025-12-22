from sqlalchemy.orm import Session
from datetime import datetime

from app.models import InventoryHistory


def get_inventory_history(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None
):
    query = db.query(InventoryHistory)

    if start_date:
        query = query.filter(InventoryHistory.timestamp >= start_date)

    if end_date:
        query = query.filter(InventoryHistory.timestamp <= end_date)

    return query.order_by(InventoryHistory.timestamp.desc()).all()
