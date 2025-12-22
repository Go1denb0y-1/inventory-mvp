from sqlalchemy.orm import Session
from datetime import datetime

from app.models import Product, InventoryHistory

def apply_inventory_change(
    *,
    db: Session,
    product_sku: str,
    delta: int,
    action: str,
    change_type: str,
    source: str
):
    # Lock product row to prevent race conditions
    product = (
        db.query(Product)
        .filter(Product.sku == product_sku)
        .with_for_update()
        .first()
    )

    if not product:
        raise ValueError("Product not found")

    quantity_before = product.quantity
    quantity_after = quantity_before + delta

    if quantity_after < 0:
        raise ValueError("Insufficient stock")

    product.quantity = quantity_after

    history = InventoryHistory(
        product_sku=product.sku,
        change_type=change_type,
        delta=delta,
        action=action,
        quantity_before=quantity_before,
        quantity_after=quantity_after,
        source=source,
        timestamp=datetime.utcnow()
    )

    db.add(history)
    db.commit()
    db.refresh(product)
    db.refresh(history)

    return {
        "sku": product.sku,
        "quantity_before": quantity_before,
        "quantity_after": quantity_after,
        "delta": delta,
        "history_id": history.id
    }
