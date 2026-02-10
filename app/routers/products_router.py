from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, case, and_, or_, desc, asc
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import time
import logging
import os
import requests
from app.database import get_db
from app.models import Product, Transaction, InventoryHistory, ChangeType, SourceType, TransactionType
from app.schemas import (
    ProductCreate, ProductOut, ProductUpdate, ProductPayload,
    ProductImport, ImportResult, SuccessResponse, ErrorResponse,
    BulkOperationResult, PaginationParams, SearchQuery
)


router = APIRouter(
    prefix="/products",
    tags=["Inventory History"]
)

logger = logging.getLogger(__name__)

# Config from environment (set these when deploying)
FRIEND_API_URL = os.getenv(
    "FRIEND_API_URL", "https://eoi-b1-1.onrender.com/api/product_connect"
)
FRIEND_API_KEY = os.getenv("FRIEND_API_KEY", "")
FRIEND_API_TIMEOUT = int(os.getenv("FRIEND_API_TIMEOUT", "10"))





class FriendAPIClient:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = FRIEND_API_TIMEOUT):
        self.base_url = (url or FRIEND_API_URL).rstrip("/")
        self.api_key = api_key or FRIEND_API_KEY
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def send_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST the product_data to the friend endpoint. Returns a dict with status and details."""
        if not self.base_url:
            return {"status": "error", "error": "Friend API URL not configured"}

        try:
            start = time.time()
            response = requests.post(self.base_url, json=product_data, headers=self.headers, timeout=self.timeout)
            elapsed = time.time() - start

            try:
                body = response.json()
            except ValueError:
                body = response.text

            return {
                "status": "success" if response.status_code in (200, 201) else "error",
                "status_code": response.status_code,
                "response_body": body,
                "time_elapsed": elapsed,
            }

        except requests.exceptions.Timeout:
            return {"status": "error", "error": f"timeout after {self.timeout}s"}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "error": "connection error"}
        except Exception as e:
            logger.exception("Unexpected error while sending to friend API")
            return {"status": "error", "error": str(e), "exception_type": type(e).__name__}


@router.post("/{sku}/sync", status_code=status.HTTP_200_OK)
def sync_product_by_sku(sku: str, payload: ProductPayload):
    """
    Sync product (payload) to friend API.
    Path sku is authoritative: payload.sku will be set to path sku.
    """

    # Enforce path-sku as authoritative
    payload.sku = sku

    # Convert model to dict and normalize last_updated
    body = payload.dict()
    if body.get("last_updated") and isinstance(body["last_updated"], datetime):
        dt = body["last_updated"]
        # ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        body["last_updated"] = dt.isoformat()

    if not FRIEND_API_URL:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Friend API URL not configured")

    client = FriendAPIClient()
    result = client.send_product(body)

    # Successful forward
    if result.get("status") == "success" and result.get("status_code") in (200, 201):
        return JSONResponse(
            status_code=200,
            content={
                "status": "forwarded",
                "friend_api_url": client.base_url,
                "friend_status_code": result.get("status_code"),
                "friend_response": result.get("response_body"),
                "time_elapsed": result.get("time_elapsed"),
            },
        )

    # Upstream returned non-success or network error
    detail = result.get("response_body") or result.get("error") or "Unknown error"
    upstream_status = result.get("status_code") if result.get("status_code") else 502

    # Map socket timeout to 504 for clarity
    if "timeout" in (result.get("error") or ""):
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail)

    raise HTTPException(status_code=upstream_status, detail=detail)


# ----------------------------
# CRUD Operations
# ----------------------------
@router.post(
    "/",
    response_model=ProductOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new product",
    description="Create a new product with validation. SKU will be converted to uppercase."
)
def create_product(
    product_data: ProductCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new product with the following validations:
    
    - SKU must be unique
    - Price must be positive if provided
    - Cost must be positive if provided
    - Quantity must be non-negative
    - RFID tag must be unique if provided
    """
    try:
        sku = product_data.sku.upper().strip()
        
        # Check if product already exists
        existing = db.query(Product).filter(Product.sku == sku).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Product with SKU '{sku}' already exists"
            )
        
        # Check if RFID tag is already in use
        if product_data.rfid_tag:
            rfid_tag = product_data.rfid_tag.upper().strip()
            existing_rfid = db.query(Product).filter(
                Product.rfid_tag == rfid_tag,
                Product.rfid_tag.isnot(None)
            ).first()
            if existing_rfid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"RFID tag '{rfid_tag}' is already assigned to product '{existing_rfid.sku}'"
                )
        
        # Validate business rules
        if product_data.quantity and product_data.quantity < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quantity cannot be negative"
            )
        
        if product_data.price is not None and product_data.price <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Price must be positive"
            )
        
        if product_data.cost is not None and product_data.cost <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cost must be positive"
            )
        
        # Validate min/max quantity
        if (product_data.min_quantity is not None and 
            product_data.max_quantity is not None and
            product_data.min_quantity > product_data.max_quantity):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Minimum quantity cannot be greater than maximum quantity"
            )
        
        # Create product
        product_dict = product_data.dict(exclude_unset=True)
        product_dict['sku'] = sku
        
        if product_dict.get('rfid_tag'):
            product_dict['rfid_tag'] = product_dict['rfid_tag'].upper()
        
        # Calculate low stock flag
        if product_dict.get('min_quantity') is not None:
            product_dict['is_low_stock'] = product_dict.get('quantity', 0) <= product_dict['min_quantity']
        else:
            product_dict['is_low_stock'] = False
        
        product = Product(**product_dict)
        
        db.add(product)
        db.commit()
        db.refresh(product)
        
        # Create inventory history entry for initial quantity
        if product.quantity > 0:
            history_entry = InventoryHistory(
                product_id=product.id,
                product_sku=product.sku,
                change_type=ChangeType.MANUAL_IN,
                quantity_before=0,
                quantity_after=product.quantity,
                source=SourceType.MANUAL,
                reason="Initial stock",
                performed_by="system"
            )
            db.add(history_entry)
            db.commit()
        
        return product
        
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database integrity error. Please check unique constraints."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating product: {str(e)}"
        )

@router.get(
    "/",
    response_model=List[ProductOut],
    summary="List all products",
    description="Retrieve products with pagination, filtering, and sorting."
)
def list_products(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    active_only: bool = Query(True, description="Show only active products"),
    low_stock_only: bool = Query(False, description="Show only low stock products"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[Decimal] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[Decimal] = Query(None, ge=0, description="Maximum price filter"),
    in_stock_only: bool = Query(False, description="Show only products with stock > 0"),
    sort_by: str = Query("name", description="Field to sort by"),
    sort_order: str = Query("asc", description="Sort order: asc or desc")
):
    """
    Get a list of products with advanced filtering and sorting options.
    """
    try:
        query = db.query(Product)
        
        # Apply filters
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
        
        # Apply sorting
        valid_sort_fields = ['sku', 'name', 'category', 'quantity', 'price', 'created_at']
        if sort_by in valid_sort_fields:
            order_func = desc if sort_order.lower() == "desc" else asc
            query = query.order_by(order_func(getattr(Product, sort_by)))
        else:
            # Default sorting
            query = query.order_by(asc(Product.name))
        
        # Apply pagination
        products = query.offset(skip).limit(limit).all()
        
        return products
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching products: {str(e)}"
        )

@router.get(
    "/{sku}",
    response_model=ProductOut,
    summary="Get product by SKU",
    description="Retrieve a single product by its SKU."
)
def get_product(
    sku: str,
    db: Session = Depends(get_db),
    include_history: bool = Query(False, description="Include recent inventory history")
):
    """
    Get detailed information about a product by SKU.
    Optionally include recent inventory history.
    """
    try:
        sku = sku.upper().strip()
        
        # Get product with optional joins
        query = db.query(Product)
        
        if include_history:
            query = query.options(
                joinedload(Product.history_entries)
                .load_only(InventoryHistory.change_type, InventoryHistory.delta, 
                          InventoryHistory.timestamp, InventoryHistory.reason)
                .order_by(desc(InventoryHistory.timestamp))
                .limit(10)
            )
        
        product = query.filter(Product.sku == sku).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with SKU '{sku}' not found"
            )
        
        if not product.is_active:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Product '{sku}' is inactive"
            )
        
        return product
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching product: {str(e)}"
        )

@router.put(
    "/{sku}",
    response_model=ProductOut,
    summary="Update product",
    description="Update an existing product. Supports partial updates."
)
def update_product(
    sku: str,
    product_data: ProductUpdate,
    db: Session = Depends(get_db),
    update_quantity: bool = Query(False, description="Track quantity change in history")
):
    """
    Update product information. Partial updates are supported.
    
    If quantity is being updated and update_quantity=True, 
    an inventory history entry will be created.
    """
    try:
        sku = sku.upper().strip()
        
        product = db.query(Product).filter(Product.sku == sku).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with SKU '{sku}' not found"
            )
        
        if not product.is_active:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Cannot update inactive product '{sku}'"
            )
        
        # Store old values for history
        old_quantity = product.quantity
        old_rfid = product.rfid_tag
        
        # Get update data
        update_dict = product_data.dict(exclude_unset=True)
        
        # Handle RFID tag changes
        if 'rfid_tag' in update_dict and update_dict['rfid_tag']:
            rfid_tag = update_dict['rfid_tag'].upper().strip()
            
            # Check if new RFID is already in use by another product
            existing_rfid = db.query(Product).filter(
                Product.rfid_tag == rfid_tag,
                Product.sku != sku,
                Product.rfid_tag.isnot(None)
            ).first()
            
            if existing_rfid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"RFID tag '{rfid_tag}' is already assigned to product '{existing_rfid.sku}'"
                )
            
            update_dict['rfid_tag'] = rfid_tag
        
        # Handle quantity updates with history tracking
        if 'quantity' in update_dict and update_quantity:
            new_quantity = update_dict['quantity']
            
            if new_quantity < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Quantity cannot be negative"
                )
            
            if new_quantity != old_quantity:
                # Create inventory history entry
                delta = new_quantity - old_quantity
                change_type = ChangeType.ADJUSTMENT if delta > 0 else ChangeType.CORRECTION
                
                history_entry = InventoryHistory(
                    product_id=product.id,
                    product_sku=product.sku,
                    change_type=change_type,
                    quantity_before=old_quantity,
                    quantity_after=new_quantity,
                    source=SourceType.MANUAL,
                    reason=f"Manual quantity adjustment",
                    performed_by="system"
                )
                db.add(history_entry)
        
        # Update product fields
        for field, value in update_dict.items():
            setattr(product, field, value)
        
        # Recalculate low stock flag if min_quantity or quantity changed
        if 'min_quantity' in update_dict or 'quantity' in update_dict:
            if product.min_quantity is not None:
                product.is_low_stock = product.quantity <= product.min_quantity
            else:
                product.is_low_stock = False
        
        # Update timestamp
        product.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(product)
        
        return product
        
    except HTTPException:
        raise
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Database integrity error. Please check unique constraints."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating product: {str(e)}"
        )

@router.delete(
    "/{sku}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete product",
    description="Delete a product by SKU. Can be soft or hard delete."
)
def delete_product(
    sku: str,
    db: Session = Depends(get_db),
    hard_delete: bool = Query(False, description="Permanently delete instead of deactivating"),
    force: bool = Query(False, description="Force delete even if product has transactions")
):
    """
    Delete a product. By default, performs a soft delete (deactivation).
    
    WARNING: Hard delete is irreversible and will remove all related data.
    """
    try:
        sku = sku.upper().strip()
        
        product = db.query(Product).filter(Product.sku == sku).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with SKU '{sku}' not found"
            )
        
        if hard_delete:
            # Check if product has transactions
            transaction_count = db.query(func.count(Transaction.id)).filter(
                Transaction.product_sku == sku
            ).scalar()
            
            if transaction_count > 0 and not force:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Cannot delete product '{sku}' because it has {transaction_count} associated transactions. Use force=true to override."
                )
            
            # Hard delete - remove all related history first
            db.query(InventoryHistory).filter(
                InventoryHistory.product_sku == sku
            ).delete(synchronize_session=False)
            
            db.delete(product)
            message = f"Product '{sku}' permanently deleted"
        else:
            # Soft delete - deactivate
            product.is_active = False
            product.updated_at = datetime.utcnow()
            message = f"Product '{sku}' deactivated"
        
        db.commit()
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting product: {str(e)}"
        )

# ----------------------------
# Search & Filter Operations
# ----------------------------
@router.get(
    "/search",
    response_model=List[ProductOut],
    summary="Search products",
    description="Search products by name, SKU, barcode, or tags."
)
def search_products(
    db: Session = Depends(get_db),
    q: str = Query(..., min_length=2, description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    include_inactive: bool = Query(False, description="Include inactive products")
):
    """
    Search products using text search across multiple fields.
    """
    try:
        search_term = f"%{q.upper()}%"
        
        query = db.query(Product).filter(
            and_(
                Product.is_active == (not include_inactive) if not include_inactive else True,
                or_(
                    Product.sku.ilike(search_term),
                    Product.name.ilike(search_term),
                    Product.barcode.ilike(search_term),
                    Product.category.ilike(search_term),
                    Product.tags.contains([q])
                )
            )
        ).order_by(asc(Product.name))
        
        products = query.offset(skip).limit(limit).all()
        
        return products
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while searching products: {str(e)}"
        )

@router.get(
    "/category/{category}",
    response_model=List[ProductOut],
    summary="Get products by category",
    description="List all products in a specific category."
)
def get_products_by_category(
    category: str,
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True)
):
    """
    Get all products in a specific category.
    """
    try:
        query = db.query(Product).filter(
            Product.category == category
        )
        
        if active_only:
            query = query.filter(Product.is_active == True)
        
        products = query.order_by(asc(Product.name)).offset(skip).limit(limit).all()
        
        return products
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching category products: {str(e)}"
        )

# ----------------------------
# Bulk Operations
# ----------------------------
@router.post(
    "/bulk",
    response_model=BulkOperationResult,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk create products",
    description="Create multiple products at once."
)
def bulk_create_products(
    products_data: List[ProductCreate],
    db: Session = Depends(get_db)
):
    """
    Create multiple products in a single operation.
    Returns success/failure count for each product.
    """
    try:
        results = []
        created_count = 0
        errors = []
        
        for index, product_data in enumerate(products_data):
            try:
                sku = product_data.sku.upper().strip()
                
                # Check for duplicates in batch
                if any(r.get('sku') == sku for r in results):
                    errors.append({
                        "index": index,
                        "sku": sku,
                        "error": f"Duplicate SKU in batch"
                    })
                    continue
                
                # Check if product already exists in database
                existing = db.query(Product).filter(Product.sku == sku).first()
                if existing:
                    errors.append({
                        "index": index,
                        "sku": sku,
                        "error": f"Product already exists"
                    })
                    continue
                
                # Validate product
                if product_data.quantity and product_data.quantity < 0:
                    errors.append({
                        "index": index,
                        "sku": sku,
                        "error": "Quantity cannot be negative"
                    })
                    continue
                
                # Create product
                product_dict = product_data.dict(exclude_unset=True)
                product_dict['sku'] = sku
                
                if product_dict.get('rfid_tag'):
                    product_dict['rfid_tag'] = product_dict['rfid_tag'].upper()
                
                product = Product(**product_dict)
                
                db.add(product)
                results.append({"sku": sku, "status": "created"})
                created_count += 1
                
            except Exception as e:
                errors.append({
                    "index": index,
                    "sku": product_data.sku if hasattr(product_data, 'sku') else "unknown",
                    "error": str(e)
                })
        
        if created_count > 0:
            db.commit()
        
        return BulkOperationResult(
            total=len(products_data),
            successful=created_count,
            failed=len(errors),
            errors=errors
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in bulk operation: {str(e)}"
        )

@router.put(
    "/bulk/update",
    response_model=BulkOperationResult,
    summary="Bulk update products",
    description="Update multiple products at once."
)
def bulk_update_products(
    updates: List[Dict[str, Any]] = Body(..., description="List of updates with sku and data"),
    db: Session = Depends(get_db)
):
    """
    Update multiple products in a single operation.
    Each update should have 'sku' and 'data' fields.
    """
    try:
        updated_count = 0
        errors = []
        
        for index, update in enumerate(updates):
            try:
                sku = update.get('sku', '').upper().strip()
                data = update.get('data', {})
                
                if not sku:
                    errors.append({
                        "index": index,
                        "sku": "unknown",
                        "error": "Missing SKU"
                    })
                    continue
                
                product = db.query(Product).filter(Product.sku == sku).first()
                
                if not product:
                    errors.append({
                        "index": index,
                        "sku": sku,
                        "error": "Product not found"
                    })
                    continue
                
                # Update fields
                for field, value in data.items():
                    if hasattr(product, field):
                        setattr(product, field, value)
                
                product.updated_at = datetime.utcnow()
                updated_count += 1
                
            except Exception as e:
                errors.append({
                    "index": index,
                    "sku": update.get('sku', 'unknown'),
                    "error": str(e)
                })
        
        if updated_count > 0:
            db.commit()
        
        return BulkOperationResult(
            total=len(updates),
            successful=updated_count,
            failed=len(errors),
            errors=errors
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in bulk update: {str(e)}"
        )

# ----------------------------
# Special Operations
# ----------------------------
@router.get(
    "/low-stock",
    response_model=List[ProductOut],
    summary="Get low stock products",
    description="List all products that are low on stock."
)
def get_low_stock_products(
    db: Session = Depends(get_db),
    threshold: Optional[int] = Query(None, ge=0, description="Custom low stock threshold"),
    include_zero: bool = Query(True, description="Include out-of-stock products")
):
    """
    Get products that are low on stock based on their min_quantity or custom threshold.
    """
    try:
        query = db.query(Product).filter(
            Product.is_active == True
        )
        
        if threshold is not None:
            # Use custom threshold
            query = query.filter(Product.quantity <= threshold)
        else:
            # Use product's min_quantity or default to quantity <= 5
            query = query.filter(
                or_(
                    and_(Product.min_quantity.isnot(None), Product.quantity <= Product.min_quantity),
                    and_(Product.min_quantity.is_(None), Product.quantity <= 5)
                )
            )
        
        if not include_zero:
            query = query.filter(Product.quantity > 0)
        
        products = query.order_by(asc(Product.quantity)).all()
        
        return products
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching low stock products: {str(e)}"
        )

@router.put(
    "/{sku}/adjust-quantity",
    response_model=ProductOut,
    summary="Adjust product quantity",
    description="Adjust product quantity with history tracking."
)
def adjust_product_quantity(
    sku: str,
    delta: int = Query(..., description="Quantity change (positive for increase, negative for decrease)"),
    reason: str = Query("Manual adjustment", description="Reason for adjustment"),
    db: Session = Depends(get_db)
):
    """
    Adjust product quantity by a specific amount.
    Creates an inventory history entry.
    """
    try:
        sku = sku.upper().strip()
        
        product = db.query(Product).filter(Product.sku == sku).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with SKU '{sku}' not found"
            )
        
        if not product.is_active:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail=f"Cannot adjust quantity for inactive product '{sku}'"
            )
        
        new_quantity = product.quantity + delta
        
        if new_quantity < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot reduce quantity below 0. Current: {product.quantity}, requested reduction: {abs(delta)}"
            )
        
        # Create inventory history
        history_entry = InventoryHistory(
            product_id=product.id,
            product_sku=product.sku,
            change_type=ChangeType.ADJUSTMENT if delta > 0 else ChangeType.CORRECTION,
            quantity_before=product.quantity,
            quantity_after=new_quantity,
            source=SourceType.MANUAL,
            reason=reason,
            performed_by="system"
        )
        db.add(history_entry)
        
        # Update product
        product.quantity = new_quantity
        
        if product.min_quantity is not None:
            product.is_low_stock = new_quantity <= product.min_quantity
        
        product.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(product)
        
        return product
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adjusting quantity: {str(e)}"
        )

@router.put(
    "/{sku}/activate",
    response_model=ProductOut,
    summary="Activate product",
    description="Activate a previously deactivated product."
)
def activate_product(
    sku: str,
    db: Session = Depends(get_db)
):
    """
    Reactivate a deactivated product.
    """
    try:
        sku = sku.upper().strip()
        
        product = db.query(Product).filter(Product.sku == sku).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with SKU '{sku}' not found"
            )
        
        if product.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Product '{sku}' is already active"
            )
        
        product.is_active = True
        product.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(product)
        
        return product
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error activating product: {str(e)}"
        )

# ----------------------------
# Statistics & Analytics
# ----------------------------
@router.get(
    "/stats/summary",
    summary="Product statistics",
    description="Get summary statistics for products."
)
def get_product_stats(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """
    Get statistical summary of products.
    """
    try:
        query = db.query(Product).filter(Product.is_active == True)
        
        if category:
            query = query.filter(Product.category == category)
        
        stats = query.with_entities(
            func.count(Product.id).label("total_products"),
            func.sum(Product.quantity).label("total_quantity"),
            func.sum(Product.quantity * Product.price).label("total_value"),
            func.avg(Product.price).label("avg_price"),
            func.count(
                case(
                    (Product.quantity == 0, 1),
                    else_=None
                )
            ).label("out_of_stock"),
            func.count(
                case(
                    (and_(Product.min_quantity.isnot(None), Product.quantity <= Product.min_quantity), 1),
                    else_=None
                )
            ).label("low_stock"),
            func.count(
                case(
                    (Product.rfid_tag.isnot(None), 1),
                    else_=None
                )
            ).label("with_rfid")
        ).first()
        
        # Categories breakdown
        categories = db.query(
            Product.category,
            func.count(Product.id).label("count"),
            func.sum(Product.quantity).label("quantity"),
            func.sum(Product.quantity * Product.price).label("value")
        ).filter(
            Product.is_active == True,
            Product.category.isnot(None)
        ).group_by(
            Product.category
        ).order_by(
            desc("count")
        ).limit(10).all()
        
        return {
            "total_products": stats.total_products or 0,
            "total_quantity": stats.total_quantity or 0,
            "total_inventory_value": stats.total_value or Decimal('0'),
            "average_price": float(stats.avg_price) if stats.avg_price else 0,
            "out_of_stock": stats.out_of_stock or 0,
            "low_stock": stats.low_stock or 0,
            "with_rfid": stats.with_rfid or 0,
            "categories": [
                {
                    "name": cat,
                    "product_count": count,
                    "total_quantity": quantity or 0,
                    "total_value": value or Decimal('0')
                }
                for cat, count, quantity, value in categories
            ]
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching statistics: {str(e)}"
        )  