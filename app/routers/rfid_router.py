from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product, InventoryHistory
from app.schemas import RFIDScan

router = APIRouter(
    prefix="/rfid",
    tags=["RFID"]
)

@router.post("/scan")
def rfid_scan(payload: RFIDScan, db: Session = Depends(get_db)):
    product = (
        db.query(Product)
        .filter(Product.sku == payload.product_sku.upper())
        .with_for_update()
        .first()
    )

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    before = product.quantity

    if payload.mode == "IN":
        delta = payload.quantity
    elif payload.mode == "OUT":
        if product.quantity < payload.quantity:
            raise HTTPException(status_code=400, detail="Insufficient stock")
        delta = -payload.quantity
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    product.quantity += delta

    history = InventoryHistory(
        product_sku=product.sku,
        change_type=f"RFID_{payload.mode}",
        delta=delta,
        quantity_before=before,
        quantity_after=product.quantity,
        source=payload.source or "RFID_READER"
    )

    db.add(history)
    db.commit()
    db.refresh(product)
    db.refresh(history)

    return {
        "message": f"RFID {payload.mode} processed",
        "sku": product.sku,
        "quantity_before": before,
        "quantity_after": product.quantity,
        "delta": delta
    }
