from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models import InventoryHistory

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)

@router.get("/movement-summary")
def movement_summary(db: Session = Depends(get_db)):
    """
    Returns net inventory movement per product SKU.
    Positive = net inflow
    Negative = net outflow
    """

    result = (
        db.query(
            InventoryHistory.product_sku,
            func.sum(InventoryHistory.delta).label("net_movement")
        )
        .group_by(InventoryHistory.product_sku)
        .all()
    )

    return [
        {
            "sku": sku,
            "net_movement": net_movement
        }
        for sku, net_movement in result
    ]
