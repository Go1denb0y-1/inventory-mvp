from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models import InventoryHistory


def get_movement_summary(db: Session):
    rows = (
        db.query(
            InventoryHistory.product_sku,
            func.sum(InventoryHistory.delta).label("net_movement")
        )
        .group_by(InventoryHistory.product_sku)
        .all()
    )

    return [
        {"sku": sku, "net_movement": movement}
        for sku, movement in rows
    ]
