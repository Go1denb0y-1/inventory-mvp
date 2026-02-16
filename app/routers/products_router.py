import os
import logging
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database import get_db
from app.models import Product
from app.schemas import ProductCreate, ProductOut, ProductPayload

router = APIRouter(prefix="/products", tags=["Products"])

# --- logging ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# --- friend api config ---
FRIEND_API_URL = os.getenv("FRIEND_API_URL", "https://eoi-b1-1.onrender.com/api/product_connect")
FRIEND_API_KEY = os.getenv("FRIEND_API_KEY", "dummy-key")
FRIEND_API_TIMEOUT = int(os.getenv("FRIEND_API_TIMEOUT", "60"))

class FriendAPIClient:
    """Small helper to POST the product payload to the Friend API.
    This method treats failures as non-fatal and returns bool success."""
    def __init__(self, url: str = FRIEND_API_URL, timeout: int = FRIEND_API_TIMEOUT):
        self.url = url
        self.headers = {
            
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def send_product(self, payload: dict) -> bool:
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            if resp.status_code in (200, 201, 202):
                return True
            logger.warning("Friend API returned status=%s body=%s", resp.status_code, resp.text)
            return False
        except requests.RequestException as exc:
            logger.exception("Friend API request failed: %s", exc)
            return False



@router.post("/", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(product_data: ProductCreate, db: Session = Depends(get_db)):
    """
    Create a product in the local DB and attempt a non-blocking sync with the Friend API.
    Returns the created Product (or raises HTTPException on validation/db error).
    """
    try:
        # normalise SKU
        sku = product_data.sku.upper().strip()
        # uniqueness check
        existing = db.query(Product).filter(Product.sku == sku).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"Product with SKU '{sku}' already exists")

        # RFID uniqueness
        if product_data.rfid_tag:
            rfid = product_data.rfid_tag.upper().strip()
            existing_rfid = db.query(Product).filter(Product.rfid_tag == rfid).first()
            if existing_rfid:
                raise HTTPException(status_code=400, detail=f"RFID tag '{rfid}' is already assigned")
            product_data.rfid_tag = rfid

        # Prepare dict for model creation
        product_dict = product_data.dict(exclude_unset=True)
        product_dict["sku"] = sku

        # Ensure numeric fields are plain floats (avoid Decimal persistence surprises)
        if "price" in product_dict and product_dict["price"] is not None:
            product_dict["price"] = float(product_dict["price"])
        if "cost" in product_dict and product_dict["cost"] is not None:
            product_dict["cost"] = float(product_dict["cost"])

        # Create and persist
        product = Product(**product_dict)
        db.add(product)
        db.commit()
        db.refresh(product)

        # Build payload for Friend API
        payload = ProductPayload(
            sku=product.sku,
            name=product.name,
            category=product.category,
            quantity=product.quantity,
            rfid_tag=product.rfid_tag,
            price=float(product.price) if product.price is not None else None,
            cost=float(product.cost) if product.cost is not None else None,
            tags=product.tags or [],
            location=product.location,
            supplier=product.supplier,
            is_active=product.is_active,
            last_updated=datetime.now(timezone.utc),
            source_system=os.getenv("SERVICE_NAME", "inventory_system"),
        )

        # Non-blocking sync (failures are logged but do not rollback the DB)
        client = FriendAPIClient()
        success = client.send_product(jsonable_encoder(payload))
        if success:
            logger.info("Friend API sync succeeded for SKU=%s", product.sku)
        else:
            logger.warning("Friend API sync failed for SKU=%s", product.sku)

        return product

    except IntegrityError as exc:
        db.rollback()
        logger.exception("Database integrity error: %s", exc)
        raise HTTPException(status_code=400, detail="Database integrity error (possible constraint violation)")
    except HTTPException:
        # re-raise validation / client errors after rolling back
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected error creating product: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")



