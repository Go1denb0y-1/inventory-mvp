import os
import time
import requests
import logging
from fastapi.encoders import jsonable_encoder
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func, desc, asc
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database import get_db
from app.models import Product, InventoryHistory, ChangeType, SourceType, Transaction
from app.schemas import ProductCreate, ProductOut, ProductUpdate, ProductPayload

router = APIRouter(prefix="/products", tags=["Products"])
logger = logging.getLogger(__name__)

FRIEND_API_URL = os.getenv(
    "FRIEND_API_URL",
    "https://eoi-b1-1.onrender.com/api/product_connect"
)
FRIEND_API_KEY = os.getenv("FRIEND_API_KEY", "")
FRIEND_API_TIMEOUT = int(os.getenv("FRIEND_API_TIMEOUT", "60"))


# ------------------------------
# Friend API client (sync)
# ------------------------------
# add this import with your other imports
from fastapi.encoders import jsonable_encoder

# ------------------------------
# Friend API client (sync, with retries)
# ------------------------------
class FriendAPIClient:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = FRIEND_API_TIMEOUT):
        self.base_url = (url or FRIEND_API_URL).rstrip("/")
        self.api_key = api_key or FRIEND_API_KEY
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def send_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send JSON payload to the friend API.
        Returns a dict describing success/error and never raises.
        This implementation includes a small retry loop to handle cold starts/timeouts.
        """
        if not self.base_url:
            return {"status": "error", "error": "Friend API URL not configured"}

        max_attempts = 3
        backoff_seconds = 2

        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug("FriendAPIClient attempt %s sending to %s payload keys=%s", attempt, self.base_url, list(product_data.keys()))
                start = time.time()
                resp = requests.post(self.base_url, json=product_data, headers=self.headers, timeout=self.timeout)
                elapsed = time.time() - start

                try:
                    body = resp.json()
                except ValueError:
                    body = resp.text

                status = "success" if resp.status_code in (200, 201) else "error"
                return {
                    "status": status,
                    "status_code": resp.status_code,
                    "response_body": body,
                    "time_elapsed": elapsed,
                    "attempt": attempt
                }

            except requests.exceptions.Timeout:
                logger.warning("FriendAPIClient timeout on attempt %s (timeout=%ss)", attempt, self.timeout)
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                return {"status": "error", "error": f"timeout after {self.timeout}s (attempts={max_attempts})"}

            except requests.exceptions.ConnectionError as ce:
                logger.warning("FriendAPIClient connection error on attempt %s: %s", attempt, str(ce))
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)
                    continue
                return {"status": "error", "error": "connection error"}

            except Exception as e:
                logger.exception("Unexpected error in FriendAPIClient on attempt %s", attempt)
                return {"status": "error", "error": str(e), "exception_type": type(e).__name__}


# ------------------------------
# CREATE PRODUCT (fixed + safe friend sync)
# ------------------------------
@router.post("/product_connect")
def test_friend_sync():
    test_payload = {
        "sku": "TEST123",
        "name": "Integration Test Product",
        "price": 1.00
    }

    friend_client = FriendAPIClient()
    response = friend_client.send_product(test_payload)

    return response

@router.post("/", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(product_data: ProductCreate, db: Session = Depends(get_db)):
    """
    Create product locally, then attempt to notify friend API.
    Friend API failures do NOT roll back the local DB transaction.
    """
    try:
        # -------------------------
        # Basic validation
        # -------------------------
        sku = product_data.sku.upper().strip()
        existing = db.query(Product).filter(Product.sku == sku).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Product with SKU '{sku}' already exists")
        if product_data.quantity is not None and product_data.quantity < 0:
            raise HTTPException(status_code=400, detail="Quantity cannot be negative")
        if product_data.price is not None and product_data.price <= 0:
            raise HTTPException(status_code=400, detail="Price must be positive")
        if product_data.cost is not None and product_data.cost <= 0:
            raise HTTPException(status_code=400, detail="Cost must be positive")

        # -------------------------
        # Filter allowed fields
        # -------------------------
        # Exclude unwanted fields explicitly
        excluded_fields = {"id", "created_at", "updated_at", "is_low_stock", "reorder_point", "max_quantity", "min_quantity", "barcode"}
        allowed_fields = {c.name for c in Product.__table__.columns if c.name not in excluded_fields}

        product_dict = {k: v for k, v in product_data.dict(exclude_unset=True).items() if k in allowed_fields}
        product_dict["sku"] = sku

        if product_dict.get("rfid_tag"):
            rfid_tag = product_dict["rfid_tag"].upper().strip()
            existing_rfid = db.query(Product).filter(Product.rfid_tag == rfid_tag).first()
            if existing_rfid:
                raise HTTPException(status_code=400, detail=f"RFID '{rfid_tag}' already assigned")
            product_dict["rfid_tag"] = rfid_tag

        # -------------------------
        # Create product
        # -------------------------
        product = Product(**product_dict)
        db.add(product)
        db.commit()
        db.refresh(product)

        # -------------------------
        # Build payload for friend API
        # -------------------------
        try:
            payload_obj = ProductPayload(
                sku=product.sku,
                name=product.name,
                category=product.category,
                quantity=int(product.quantity or 0),
                rfid_tag=product.rfid_tag,
                price=float(product.price) if product.price is not None else None,
                cost=float(product.cost) if product.cost is not None else None,
                tags=product.tags or [],
                location=product.location,
                supplier=product.supplier,
                is_active=product.is_active,
                last_updated=datetime.now(timezone.utc),  # simplified
                source_system=os.getenv("SERVICE_NAME", "inventory_system")
            )

            encoded_payload = jsonable_encoder(payload_obj)
            friend_client = FriendAPIClient()
            friend_response = friend_client.send_product(encoded_payload)

            if friend_response.get("status") != "success":
                logger.warning("Friend API sync failed for SKU=%s: %s", product.sku, friend_response)
            else:
                logger.info("Friend API sync success for SKU=%s", product.sku)

        except Exception:
            logger.exception("Failed to send product to friend API; skipping sync")

        # -------------------------
        # Create initial inventory history
        # -------------------------
        if product.quantity and product.quantity > 0:
            history = InventoryHistory(
                product_id=product.id,
                product_sku=product.sku,
                change_type=ChangeType.MANUAL_IN,
                quantity_before=0,
                quantity_after=product.quantity,
                source=SourceType.MANUAL,
                reason="Initial stock",
                performed_by="system"
            )
            db.add(history)
            db.commit()

        return product

    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database integrity error")
    except Exception:
        db.rollback()
        logger.exception("Error creating product")
        raise HTTPException(status_code=500, detail="Internal server error")

# -------------------------------------------------------------------------
# (remaining routes unchanged — paste the rest of your file below)
# -------------------------------------------------------------------------
# ... list_products, get_product, update_product, delete_product (unchanged)

@router.get("/", response_model=List[ProductOut])
def list_products(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = True,
    low_stock_only: bool = False,
    category: Optional[str] = None,
    min_price: Optional[Decimal] = None,
    max_price: Optional[Decimal] = None,
    in_stock_only: bool = False,
    sort_by: str = "name",
    sort_order: str = "asc"
):

    try:
        query = db.query(Product)

        if active_only:
            query = query.filter(Product.is_active == True)

        if low_stock_only:
            query = query.filter(Product.is_low_stock == True)

        if category:
            query = query.filter(Product.category == category)

        if min_price is not None:
            query = query.filter(Product.price >= min_price)

        if max_price is not None:
            query = query.filter(Product.price <= max_price)

        if in_stock_only:
            query = query.filter(Product.quantity > 0)

        valid_sort_fields = ["sku", "name", "category", "quantity", "price", "created_at"]

        if sort_by in valid_sort_fields:
            order_func = desc if sort_order.lower() == "desc" else asc
            query = query.order_by(order_func(getattr(Product, sort_by)))
        else:
            query = query.order_by(asc(Product.name))

        return query.offset(skip).limit(limit).all()

    except SQLAlchemyError:
        logger.exception("Error listing products")
        raise HTTPException(500, "Database error")

@router.get("/{sku}", response_model=ProductOut)
def get_product(
    sku: str,
    db: Session = Depends(get_db),
    include_history: bool = False
):

    sku = sku.upper().strip()

    product = db.query(Product).filter(Product.sku == sku).first()

    if not product:
        raise HTTPException(404, f"Product '{sku}' not found")

    if not product.is_active:
        raise HTTPException(410, f"Product '{sku}' is inactive")

    if include_history:
        history = (
            db.query(InventoryHistory)
            .filter(InventoryHistory.product_id == product.id)
            .order_by(desc(InventoryHistory.timestamp))
            .limit(10)
            .all()
        )
        product.history_entries = history

    return product

@router.put("/{sku}", response_model=ProductOut)
def update_product(
    sku: str,
    product_data: ProductUpdate,
    db: Session = Depends(get_db),
    update_quantity: bool = False
):

    sku = sku.upper().strip()
    product = db.query(Product).filter(Product.sku == sku).first()

    if not product:
        raise HTTPException(404, "Product not found")

    if not product.is_active:
        raise HTTPException(410, "Product inactive")

    old_quantity = product.quantity
    update_dict = product_data.dict(exclude_unset=True)

    if "sku" in update_dict:
        update_dict.pop("sku")

    if "quantity" in update_dict:
        new_quantity = update_dict["quantity"]

        if new_quantity < 0:
            raise HTTPException(400, "Quantity cannot be negative")

        if update_quantity and new_quantity != old_quantity:
            delta = new_quantity - old_quantity

            history = InventoryHistory(
                product_id=product.id,
                product_sku=product.sku,
                change_type=ChangeType.ADJUSTMENT if delta > 0 else ChangeType.CORRECTION,
                quantity_before=old_quantity,
                quantity_after=new_quantity,
                source=SourceType.MANUAL,
                reason="Manual adjustment",
                performed_by="system"
            )
            db.add(history)

    for field, value in update_dict.items():
        setattr(product, field, value)

    if product.min_quantity is not None:
        product.is_low_stock = product.quantity <= product.min_quantity
    else:
        product.is_low_stock = False

    product.updated_at = datetime.now(timezone.utc)

    try:
        db.commit()
        db.refresh(product)
        return product
    except Exception:
        db.rollback()
        logger.exception("Error updating product")
        raise HTTPException(500, "Internal server error")

@router.delete("/{sku}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(
    sku: str,
    db: Session = Depends(get_db),
    hard_delete: bool = False,
    force: bool = False
):

    sku = sku.upper().strip()
    product = db.query(Product).filter(Product.sku == sku).first()

    if not product:
        raise HTTPException(404, "Product not found")

    try:
        if hard_delete:
            transaction_count = db.query(func.count(Transaction.id)).filter(
                Transaction.product_sku == sku
            ).scalar()

            if transaction_count > 0 and not force:
                raise HTTPException(
                    409,
                    f"Product has {transaction_count} transactions. Use force=true."
                )

            db.query(InventoryHistory).filter(
                InventoryHistory.product_sku == sku
            ).delete(synchronize_session=False)

            db.delete(product)

        else:
            product.is_active = False
            product.updated_at = datetime.now(timezone.utc)

        db.commit()

    except HTTPException:
        raise
    except Exception:
        db.rollback()
        logger.exception("Error deleting product")
        raise HTTPException(500, "Internal server error")

    return JSONResponse(status_code=204, content=None)
