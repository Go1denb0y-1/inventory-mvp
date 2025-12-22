from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product, Transaction, InventoryHistory
from app.schemas import TransactionCreate
from datetime import datetime

router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"]
)

@router.post("/")
def create_transaction(payload: TransactionCreate, db: Session = Depends(get_db)):
    product = (
        db.query(Product)
        .filter(Product.sku == payload.product_sku.upper())
        .with_for_update()
        .first()
    )

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    before = product.quantity

    if payload.transaction_type == "IN":
        delta = payload.quantity
    elif payload.transaction_type == "OUT":
        if product.quantity < payload.quantity:
            raise HTTPException(status_code=400, detail="Insufficient stock")
        delta = -payload.quantity
    else:
        raise HTTPException(status_code=400, detail="Invalid transaction type")

    product.quantity += delta

    transaction = Transaction(
        product_sku=product.sku,
        transaction_type=payload.transaction_type,
        quantity=payload.quantity,
        note=payload.note,
        source=payload.source or "MANUAL"
    )

    history = InventoryHistory(
        product_sku=product.sku,
        change_type=f"TRANSACTION_{payload.transaction_type}",
        delta=delta,
        quantity_before=before,
        quantity_after=product.quantity,
        source=payload.source or "MANUAL"
    )

    db.add(transaction)
    db.add(history)
    db.commit()
    db.refresh(product)
    db.refresh(transaction)
    db.refresh(history)

    return {
        "message": "Transaction recorded",
        "sku": product.sku,
        "quantity_before": before,
        "quantity_after": product.quantity,
        "delta": delta
    }

@router.get("/")
def list_transactions(
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(Transaction)

    if start_date:
        query = query.filter(Transaction.timestamp >= start_date)
    if end_date:
        query = query.filter(Transaction.timestamp <= end_date)

    return query.order_by(Transaction.timestamp.desc()).all()