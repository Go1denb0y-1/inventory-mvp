import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import func, and_, or_

from app.database import get_db, get_db_context
from app.models import (
    Product, InventoryHistory, Transaction,
    TransactionType, ChangeType, SourceType
)
from app.schemas import (
    RFIDScan, RFIDTagCreate, RFIDTagOut,
    SuccessResponse, ErrorResponse, BulkOperationResult
)

router = APIRouter(
    prefix="/rfid",
    tags=["RFID"]
)

# ----------------------------
# RFID Scan Processing
# ----------------------------
async def process_rfid_scan_async(
    rfid_tag: str,
    mode: TransactionType,
    db: Session,
    quantity: int = 1,
    source: SourceType = SourceType.RFID,
    device_id: Optional[str] = None,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async function to process RFID scan with proper transaction handling.
    """
    with get_db_context() as session:
        try:
            # Use SELECT FOR UPDATE to prevent race conditions
            product = (
                session.query(Product)
                .filter(
                    and_(
                        Product.rfid_tag == rfid_tag.upper(),
                        Product.is_active == True
                    )
                )
                .with_for_update()
                .first()
            )
            
            if not product:
                raise ValueError(f"No active product found with RFID tag: {rfid_tag}")
            
            before_quantity = product.quantity
            
            # Process the scan
            if mode == TransactionType.IN:
                delta = quantity
                new_quantity = before_quantity + delta
                transaction_type = TransactionType.IN
                change_type = ChangeType.RFID_IN
                
            elif mode == TransactionType.OUT:
                if before_quantity < quantity:
                    raise ValueError(
                        f"Insufficient stock. Available: {before_quantity}, Requested: {quantity}"
                    )
                delta = -quantity
                new_quantity = before_quantity + delta
                transaction_type = TransactionType.OUT
                change_type = ChangeType.RFID_OUT
                
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Update product quantity
            product.quantity = new_quantity
            product.updated_at = datetime.utcnow()
            
            # Update low stock flag
            if product.min_quantity is not None:
                product.is_low_stock = product.quantity <= product.min_quantity
            
            # Create inventory history record
            history = InventoryHistory(
                product_id=product.id,
                product_sku=product.sku,
                change_type=change_type,
                quantity_before=before_quantity,
                quantity_after=new_quantity,
                source=source,
                reason=f"RFID scan: {mode.value}",
                performed_by="RFID_SYSTEM"
            )
            session.add(history)
            
            # Create transaction record
            transaction = Transaction(
                product_id=product.id,
                product_sku=product.sku,
                transaction_type=transaction_type,
                quantity=quantity,
                unit_price=product.price,
                total_value=product.price * quantity if product.price else None,
                source=source,
                device_id=device_id,
                note=f"RFID scan: {rfid_tag}"
            )
            session.add(transaction)
            
            # Update RFID tag last seen
            if device_id:
                product.rfid_tag_last_seen = datetime.utcnow()
                product.rfid_tag_device = device_id
            
            # Commit the transaction
            session.commit()
            
            return {
                "success": True,
                "rfid_tag": rfid_tag,
                "product_sku": product.sku,
                "product_name": product.name,
                "mode": mode.value,
                "quantity": quantity,
                "before_quantity": before_quantity,
                "after_quantity": new_quantity,
                "delta": delta,
                "transaction_id": transaction.id,
                "history_id": history.id
            }
            
        except ValueError as e:
            session.rollback()
            raise e
        except IntegrityError as e:
            session.rollback()
            raise ValueError(f"Database integrity error: {str(e)}")
        except Exception as e:
            session.rollback()
            raise ValueError(f"Unexpected error: {str(e)}")

@router.post("/scan", status_code=status.HTTP_202_ACCEPTED)
async def process_rfid_scan(
    payload: RFIDScan,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process an RFID scan.
    
    This endpoint accepts RFID scan data and processes it asynchronously.
    Returns immediately with an acknowledgement, processing happens in background.
    """
    try:
        # Validate RFID tag format
        if not payload.rfid_tag or not payload.rfid_tag.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="RFID tag cannot be empty"
            )
        
        # Clean and validate RFID tag
        rfid_tag = payload.rfid_tag.strip().upper()
        if len(rfid_tag) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="RFID tag cannot exceed 100 characters"
            )
        
        # Validate quantity
        if payload.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quantity must be greater than 0"
            )
        
        # Add to background tasks for async processing
        background_tasks.add_task(
            process_rfid_scan_async,
            rfid_tag=rfid_tag,
            mode=payload.mode,
            quantity=payload.quantity,
            source=payload.source,
            device_id=payload.device_id,
            location=payload.location,
            db=db
        )
        
        return {
            "status": "accepted",
            "message": "RFID scan is being processed",
            "scan_id": f"RFID-{datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}",
            "rfid_tag": rfid_tag,
            "mode": payload.mode.value,
            "quantity": payload.quantity
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RFID scan: {str(e)}"
        )

@router.post("/scan/sync")
def process_rfid_scan_sync(
    payload: RFIDScan,
    db: Session = Depends(get_db)
):
    """
    Synchronous RFID scan processing.
    
    Use this endpoint when you need immediate feedback about the scan result.
    """
    try:
        # Validate RFID tag
        if not payload.rfid_tag or not payload.rfid_tag.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="RFID tag cannot be empty"
            )
        
        rfid_tag = payload.rfid_tag.strip().upper()
        
        # Validate quantity
        if payload.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quantity must be greater than 0"
            )
        
        # Process the scan synchronously
        result = asyncio.run(
            process_rfid_scan_async(
                rfid_tag=rfid_tag,
                mode=payload.mode,
                quantity=payload.quantity,
                source=payload.source,
                device_id=payload.device_id,
                location=payload.location,
                db=db
            )
        )
        
        return {
            "success": True,
            "message": f"RFID {payload.mode.value} processed successfully",
            **result
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RFID scan: {str(e)}"
        )

@router.post("/scan/bulk", status_code=status.HTTP_202_ACCEPTED)
async def process_bulk_rfid_scans(
    scans: List[RFIDScan],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Process multiple RFID scans in bulk.
    
    Each scan is processed independently. Failures in one scan don't affect others.
    """
    try:
        if not scans:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No scans provided"
            )
        
        if len(scans) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 1000 scans allowed per request"
            )
        
        results = []
        failed_scans = []
        
        # Process each scan
        for i, scan in enumerate(scans):
            try:
                # Validate RFID tag
                if not scan.rfid_tag or not scan.rfid_tag.strip():
                    failed_scans.append({
                        "index": i,
                        "rfid_tag": scan.rfid_tag,
                        "error": "RFID tag cannot be empty"
                    })
                    continue
                
                rfid_tag = scan.rfid_tag.strip().upper()
                
                # Validate quantity
                if scan.quantity <= 0:
                    failed_scans.append({
                        "index": i,
                        "rfid_tag": rfid_tag,
                        "error": "Quantity must be greater than 0"
                    })
                    continue
                
                # Add to background tasks
                background_tasks.add_task(
                    process_rfid_scan_async,
                    rfid_tag=rfid_tag,
                    mode=scan.mode,
                    quantity=scan.quantity,
                    source=scan.source,
                    device_id=scan.device_id,
                    location=scan.location,
                    db=db
                )
                
                results.append({
                    "index": i,
                    "rfid_tag": rfid_tag,
                    "status": "queued"
                })
                
            except Exception as e:
                failed_scans.append({
                    "index": i,
                    "rfid_tag": scan.rfid_tag,
                    "error": str(e)
                })
        
        return BulkOperationResult(
            total=len(scans),
            successful=len(results),
            failed=len(failed_scans),
            errors=failed_scans
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing bulk RFID scans: {str(e)}"
        )

# ----------------------------
# RFID Tag Management
# ----------------------------
@router.post("/tags", response_model=RFIDTagOut)
def create_rfid_tag(
    payload: RFIDTagCreate,
    db: Session = Depends(get_db)
):
    """
    Associate an RFID tag with a product.
    """
    try:
        # Clean RFID tag
        rfid_tag = payload.tag_uid.strip().upper()
        
        # Check if RFID tag already exists
        existing_product = db.query(Product).filter(
            Product.rfid_tag == rfid_tag
        ).first()
        
        if existing_product:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"RFID tag {rfid_tag} is already assigned to product {existing_product.sku}"
            )
        
        # Find the product
        product = db.query(Product).filter(
            Product.sku == payload.product_sku.upper()
        ).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product with SKU {payload.product_sku} not found"
            )
        
        # Check if product already has an RFID tag
        if product.rfid_tag:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Product {product.sku} already has RFID tag {product.rfid_tag}"
            )
        
        # Assign RFID tag to product
        product.rfid_tag = rfid_tag
        product.updated_at = datetime.utcnow()
        
        # Create history entry
        history = InventoryHistory(
            product_id=product.id,
            product_sku=product.sku,
            change_type=ChangeType.ADJUSTMENT,
            quantity_before=product.quantity,
            quantity_after=product.quantity,
            source=SourceType.SYSTEM,
            reason=f"RFID tag assigned: {rfid_tag}",
            performed_by="API"
        )
        db.add(history)
        
        db.commit()
        db.refresh(product)
        
        return {
            "id": product.id,  # This would be different if you had a separate RFIDTag model
            "tag_uid": rfid_tag,
            "product_sku": product.sku,
            "product_name": product.name,
            "is_active": product.is_active,
            "first_seen": datetime.utcnow(),
            "location": payload.location,
            "notes": payload.notes
        }
        
    except HTTPException:
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="RFID tag assignment failed due to database constraint"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error assigning RFID tag: {str(e)}"
        )

@router.put("/tags/{rfid_tag}")
def update_rfid_tag(
    rfid_tag: str,
    payload: RFIDTagCreate,
    db: Session = Depends(get_db)
):
    """
    Update or reassign an RFID tag.
    """
    try:
        rfid_tag_clean = rfid_tag.strip().upper()
        
        # Find product with this RFID tag
        current_product = db.query(Product).filter(
            Product.rfid_tag == rfid_tag_clean
        ).first()
        
        if not current_product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"RFID tag {rfid_tag_clean} not found"
            )
        
        # If reassigning to a different product
        if current_product.sku != payload.product_sku.upper():
            # Find new product
            new_product = db.query(Product).filter(
                Product.sku == payload.product_sku.upper()
            ).first()
            
            if not new_product:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"New product with SKU {payload.product_sku} not found"
                )
            
            # Check if new product already has an RFID tag
            if new_product.rfid_tag:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"New product {new_product.sku} already has RFID tag {new_product.rfid_tag}"
                )
            
            # Remove from current product
            current_product.rfid_tag = None
            current_product.updated_at = datetime.utcnow()
            
            # Assign to new product
            new_product.rfid_tag = rfid_tag_clean
            new_product.updated_at = datetime.utcnow()
            
            # Create history entries
            history_current = InventoryHistory(
                product_id=current_product.id,
                product_sku=current_product.sku,
                change_type=ChangeType.ADJUSTMENT,
                quantity_before=current_product.quantity,
                quantity_after=current_product.quantity,
                source=SourceType.SYSTEM,
                reason=f"RFID tag removed: {rfid_tag_clean}",
                performed_by="API"
            )
            db.add(history_current)
            
            history_new = InventoryHistory(
                product_id=new_product.id,
                product_sku=new_product.sku,
                change_type=ChangeType.ADJUSTMENT,
                quantity_before=new_product.quantity,
                quantity_after=new_product.quantity,
                source=SourceType.SYSTEM,
                reason=f"RFID tag assigned: {rfid_tag_clean}",
                performed_by="API"
            )
            db.add(history_new)
            
        else:
            # Just update location/notes for existing assignment
            current_product.updated_at = datetime.utcnow()
        
        db.commit()
        
        return SuccessResponse(
            message=f"RFID tag {rfid_tag_clean} updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating RFID tag: {str(e)}"
        )

@router.delete("/tags/{rfid_tag}")
def remove_rfid_tag(
    rfid_tag: str,
    db: Session = Depends(get_db)
):
    """
    Remove RFID tag association from a product.
    """
    try:
        rfid_tag_clean = rfid_tag.strip().upper()
        
        # Find product with this RFID tag
        product = db.query(Product).filter(
            Product.rfid_tag == rfid_tag_clean
        ).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"RFID tag {rfid_tag_clean} not found"
            )
        
        # Remove RFID tag
        product.rfid_tag = None
        product.updated_at = datetime.utcnow()
        
        # Create history entry
        history = InventoryHistory(
            product_id=product.id,
            product_sku=product.sku,
            change_type=ChangeType.ADJUSTMENT,
            quantity_before=product.quantity,
            quantity_after=product.quantity,
            source=SourceType.SYSTEM,
            reason=f"RFID tag removed: {rfid_tag_clean}",
            performed_by="API"
        )
        db.add(history)
        
        db.commit()
        
        return SuccessResponse(
            message=f"RFID tag {rfid_tag_clean} removed from product {product.sku}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error removing RFID tag: {str(e)}"
        )

# ----------------------------
# RFID Tag Lookup & Search
# ----------------------------
@router.get("/tags/{rfid_tag}", response_model=RFIDTagOut)
def get_rfid_tag_info(
    rfid_tag: str,
    db: Session = Depends(get_db)
):
    """
    Get information about an RFID tag and its associated product.
    """
    try:
        rfid_tag_clean = rfid_tag.strip().upper()
        
        product = db.query(Product).filter(
            Product.rfid_tag == rfid_tag_clean
        ).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"RFID tag {rfid_tag_clean} not found"
            )
        
        # Get last scan time for this RFID tag
        last_transaction = db.query(Transaction).filter(
            Transaction.product_sku == product.sku,
            Transaction.source == SourceType.RFID
        ).order_by(
            Transaction.timestamp.desc()
        ).first()
        
        return {
            "id": product.id,
            "tag_uid": rfid_tag_clean,
            "product_sku": product.sku,
            "product_name": product.name,
            "product_category": product.category,
            "current_quantity": product.quantity,
            "is_active": product.is_active,
            "last_seen": last_transaction.timestamp if last_transaction else None,
            "first_seen": product.created_at,  # Approximation
            "location": product.location,
            "device_id": getattr(product, 'rfid_tag_device', None)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching RFID tag info: {str(e)}"
        )

@router.get("/tags/search")
def search_rfid_tags(
    db: Session = Depends(get_db),
    tag_prefix: Optional[str] = Query(None, description="Search by RFID tag prefix"),
    product_sku: Optional[str] = Query(None, description="Search by product SKU"),
    product_name: Optional[str] = Query(None, description="Search by product name"),
    active_only: bool = Query(True, description="Only show active products"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Search for RFID tags with various filters.
    """
    try:
        query = db.query(Product).filter(
            Product.rfid_tag.isnot(None)
        )
        
        if active_only:
            query = query.filter(Product.is_active == True)
        
        if tag_prefix:
            query = query.filter(Product.rfid_tag.startswith(tag_prefix.upper()))
        
        if product_sku:
            query = query.filter(Product.sku.ilike(f"%{product_sku.upper()}%"))
        
        if product_name:
            query = query.filter(Product.name.ilike(f"%{product_name}%"))
        
        total_count = query.count()
        products = query.offset(skip).limit(limit).all()
        
        # Get last seen times
        rfid_tags = []
        for product in products:
            last_transaction = db.query(Transaction).filter(
                Transaction.product_sku == product.sku,
                Transaction.source == SourceType.RFID
            ).order_by(
                Transaction.timestamp.desc()
            ).first()
            
            rfid_tags.append({
                "tag_uid": product.rfid_tag,
                "product_sku": product.sku,
                "product_name": product.name,
                "product_category": product.category,
                "current_quantity": product.quantity,
                "is_active": product.is_active,
                "last_seen": last_transaction.timestamp if last_transaction else None,
                "first_seen": product.created_at,
                "location": product.location
            })
        
        return {
            "total_count": total_count,
            "returned_count": len(rfid_tags),
            "rfid_tags": rfid_tags,
            "pagination": {
                "skip": skip,
                "limit": limit
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching RFID tags: {str(e)}"
        )

# ----------------------------
# RFID System Status & Diagnostics
# ----------------------------
@router.get("/status")
def get_rfid_system_status(
    db: Session = Depends(get_db)
):
    """
    Get RFID system status and statistics.
    """
    try:
        # Total RFID tags
        total_rfid_tags = db.query(func.count(Product.id)).filter(
            Product.rfid_tag.isnot(None)
        ).scalar() or 0
        
        # Active RFID tags
        active_rfid_tags = db.query(func.count(Product.id)).filter(
            Product.rfid_tag.isnot(None),
            Product.is_active == True
        ).scalar() or 0
        
        # RFID transactions today
        today = datetime.utcnow().date()
        rfid_transactions_today = db.query(func.count(Transaction.id)).filter(
            Transaction.source == SourceType.RFID,
            func.date(Transaction.timestamp) == today
        ).scalar() or 0
        
        # Products without RFID tags
        products_without_rfid = db.query(func.count(Product.id)).filter(
            Product.rfid_tag.is_(None),
            Product.is_active == True
        ).scalar() or 0
        
        # Recent RFID activity
        recent_activity = db.query(
            Transaction.timestamp,
            Transaction.product_sku,
            Transaction.transaction_type,
            Transaction.quantity,
            Product.name
        ).join(
            Product, Transaction.product_sku == Product.sku
        ).filter(
            Transaction.source == SourceType.RFID
        ).order_by(
            Transaction.timestamp.desc()
        ).limit(10).all()
        
        return {
            "status": "online",
            "timestamp": datetime.utcnow(),
            "statistics": {
                "total_rfid_tags": total_rfid_tags,
                "active_rfid_tags": active_rfid_tags,
                "inactive_rfid_tags": total_rfid_tags - active_rfid_tags,
                "products_without_rfid": products_without_rfid,
                "rfid_transactions_today": rfid_transactions_today,
                "rfid_coverage_percentage": (
                    (active_rfid_tags / (active_rfid_tags + products_without_rfid) * 100)
                    if (active_rfid_tags + products_without_rfid) > 0 else 0
                )
            },
            "recent_activity": [
                {
                    "timestamp": activity.timestamp,
                    "product_sku": activity.product_sku,
                    "product_name": activity.name,
                    "transaction_type": activity.transaction_type,
                    "quantity": activity.quantity
                }
                for activity in recent_activity
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching RFID system status: {str(e)}"
        )