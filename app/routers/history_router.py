from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.models import InventoryHistory, Product, Transaction, ChangeType, SourceType
from app.schemas import (
    InventoryHistoryOut, InventoryHistoryFilter, PaginationParams,
    SuccessResponse, ErrorResponse
)

router = APIRouter(
    prefix="/history",
    tags=["Inventory History"]
)

# ----------------------------
# CRUD Operations
# ----------------------------
@router.get("/", response_model=List[InventoryHistoryOut])
def get_inventory_history(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU"),
    change_type: Optional[ChangeType] = Query(None, description="Filter by change type"),
    source: Optional[SourceType] = Query(None, description="Filter by source"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    reason: Optional[str] = Query(None, description="Filter by reason (partial match)"),
    performed_by: Optional[str] = Query(None, description="Filter by user who performed the change"),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'")
):
    """
    Retrieve inventory history with advanced filtering and pagination.
    
    Supports filtering by:
    - Product SKU
    - Change type
    - Source
    - Date range
    - Reason (text search)
    - Performed by user
    
    Results are paginated and can be sorted by any field.
    """
    try:
        # Build query
        query = db.query(InventoryHistory).options(
            joinedload(InventoryHistory.product)
        )
        
        # Apply filters
        filters = []
        
        if product_sku:
            filters.append(InventoryHistory.product_sku == product_sku.upper())
        
        if change_type:
            filters.append(InventoryHistory.change_type == change_type)
        
        if source:
            filters.append(InventoryHistory.source == source)
        
        if start_date:
            filters.append(InventoryHistory.timestamp >= start_date)
        
        if end_date:
            filters.append(InventoryHistory.timestamp <= end_date)
        
        if reason:
            filters.append(InventoryHistory.reason.ilike(f"%{reason}%"))
        
        if performed_by:
            filters.append(InventoryHistory.performed_by.ilike(f"%{performed_by}%"))
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Apply sorting
        if hasattr(InventoryHistory, sort_by):
            order_func = desc if sort_order.lower() == "desc" else asc
            query = query.order_by(order_func(getattr(InventoryHistory, sort_by)))
        else:
            # Default sorting by timestamp descending
            query = query.order_by(desc(InventoryHistory.timestamp))
        
        # Apply pagination
        total_count = query.count()
        history = query.offset(skip).limit(limit).all()
        
        # Enrich with product names if not already loaded
        for entry in history:
            if entry.product and not hasattr(entry, 'product_name'):
                entry.product_name = entry.product.name
        
        return history
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching history: {str(e)}"
        )

@router.get("/search", response_model=List[InventoryHistoryOut])
def search_inventory_history(
    db: Session = Depends(get_db),
    q: str = Query(..., min_length=1, description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Search inventory history by product SKU, name, reason, or change type.
    """
    try:
        query = db.query(InventoryHistory).options(
            joinedload(InventoryHistory.product)
        ).filter(
            or_(
                InventoryHistory.product_sku.ilike(f"%{q}%"),
                InventoryHistory.reason.ilike(f"%{q}%"),
                InventoryHistory.change_type.ilike(f"%{q}%"),
                InventoryHistory.performed_by.ilike(f"%{q}%")
            )
        ).order_by(
            desc(InventoryHistory.timestamp)
        )
        
        total_count = query.count()
        results = query.offset(skip).limit(limit).all()
        
        # Also search by product name if we have products
        if len(results) < limit:
            product_name_query = db.query(InventoryHistory).options(
                joinedload(InventoryHistory.product)
            ).join(
                Product, InventoryHistory.product_sku == Product.sku
            ).filter(
                Product.name.ilike(f"%{q}%")
            ).order_by(
                desc(InventoryHistory.timestamp)
            ).offset(skip).limit(limit - len(results)).all()
            
            # Combine results, removing duplicates
            existing_ids = {r.id for r in results}
            for entry in product_name_query:
                if entry.id not in existing_ids:
                    results.append(entry)
                    existing_ids.add(entry.id)
        
        return results
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while searching history: {str(e)}"
        )

@router.get("/{history_id}", response_model=InventoryHistoryOut)
def get_history_entry(
    history_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific inventory history entry by ID.
    """
    try:
        history_entry = db.query(InventoryHistory).options(
            joinedload(InventoryHistory.product)
        ).filter(
            InventoryHistory.id == history_id
        ).first()
        
        if not history_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory history entry with ID {history_id} not found"
            )
        
        return history_entry
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching history entry: {str(e)}"
        )

@router.delete("/{history_id}", response_model=SuccessResponse)
def delete_history_entry(
    history_id: int,
    db: Session = Depends(get_db),
    force: bool = Query(False, description="Force delete without validation")
):
    """
    Delete an inventory history entry.
    
    WARNING: This is a destructive operation. By default, only entries
    older than 30 days can be deleted unless force=True.
    """
    try:
        history_entry = db.query(InventoryHistory).filter(
            InventoryHistory.id == history_id
        ).first()
        
        if not history_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inventory history entry with ID {history_id} not found"
            )
        
        # Prevent deletion of recent entries unless forced
        if not force:
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            if history_entry.timestamp > thirty_days_ago:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot delete history entries less than 30 days old. Use force=true to override."
                )
        
        db.delete(history_entry)
        db.commit()
        
        return SuccessResponse(
            message=f"Inventory history entry {history_id} deleted successfully"
        )
        
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while deleting history entry: {str(e)}"
        )

# ----------------------------
# Analytics & Reports
# ----------------------------
@router.get("/summary/daily")
def get_daily_history_summary(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU")
):
    """
    Get daily summary of inventory changes.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            func.date(InventoryHistory.timestamp).label("date"),
            func.count(InventoryHistory.id).label("total_changes"),
            func.sum(
                case(
                    (InventoryHistory.delta > 0, InventoryHistory.delta),
                    else_=0
                )
            ).label("total_increase"),
            func.sum(
                case(
                    (InventoryHistory.delta < 0, -InventoryHistory.delta),
                    else_=0
                )
            ).label("total_decrease"),
            func.count(
                case(
                    (InventoryHistory.change_type == ChangeType.MANUAL_IN, 1),
                    else_=None
                )
            ).label("manual_in_count"),
            func.count(
                case(
                    (InventoryHistory.change_type == ChangeType.MANUAL_OUT, 1),
                    else_=None
                )
            ).label("manual_out_count"),
            func.count(
                case(
                    (InventoryHistory.change_type == ChangeType.RFID_IN, 1),
                    else_=None
                )
            ).label("rfid_in_count"),
            func.count(
                case(
                    (InventoryHistory.change_type == ChangeType.RFID_OUT, 1),
                    else_=None
                )
            ).label("rfid_out_count")
        ).filter(
            InventoryHistory.timestamp >= cutoff_date
        )
        
        if product_sku:
            query = query.filter(InventoryHistory.product_sku == product_sku.upper())
        
        results = query.group_by(
            func.date(InventoryHistory.timestamp)
        ).order_by(
            func.date(InventoryHistory.timestamp).desc()
        ).all()
        
        return [
            {
                "date": row.date,
                "total_changes": row.total_changes or 0,
                "total_increase": row.total_increase or 0,
                "total_decrease": row.total_decrease or 0,
                "net_change": (row.total_increase or 0) - (row.total_decrease or 0),
                "change_types": {
                    "manual_in": row.manual_in_count or 0,
                    "manual_out": row.manual_out_count or 0,
                    "rfid_in": row.rfid_in_count or 0,
                    "rfid_out": row.rfid_out_count or 0,
                    "other": (row.total_changes or 0) - 
                            (row.manual_in_count or 0) - 
                            (row.manual_out_count or 0) - 
                            (row.rfid_in_count or 0) - 
                            (row.rfid_out_count or 0)
                }
            }
            for row in results
        ]
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while generating daily summary: {str(e)}"
        )

@router.get("/summary/by-product")
def get_history_summary_by_product(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    limit: int = Query(20, ge=1, le=100, description="Number of products to return")
):
    """
    Get inventory change summary grouped by product.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        results = db.query(
            InventoryHistory.product_sku,
            Product.name,
            func.count(InventoryHistory.id).label("change_count"),
            func.sum(
                case(
                    (InventoryHistory.delta > 0, InventoryHistory.delta),
                    else_=0
                )
            ).label("total_increase"),
            func.sum(
                case(
                    (InventoryHistory.delta < 0, -InventoryHistory.delta),
                    else_=0
                )
            ).label("total_decrease"),
            func.min(InventoryHistory.timestamp).label("first_change"),
            func.max(InventoryHistory.timestamp).label("last_change"),
            func.count(
                case(
                    (InventoryHistory.source == SourceType.RFID, 1),
                    else_=None
                )
            ).label("rfid_changes"),
            func.count(
                case(
                    (InventoryHistory.source == SourceType.MANUAL, 1),
                    else_=None
                )
            ).label("manual_changes")
        ).outerjoin(
            Product, InventoryHistory.product_sku == Product.sku
        ).filter(
            InventoryHistory.timestamp >= cutoff_date
        ).group_by(
            InventoryHistory.product_sku,
            Product.name
        ).order_by(
            desc("change_count")
        ).limit(limit).all()
        
        return [
            {
                "sku": row.product_sku,
                "product_name": row.name,
                "change_count": row.change_count or 0,
                "total_increase": row.total_increase or 0,
                "total_decrease": row.total_decrease or 0,
                "net_change": (row.total_increase or 0) - (row.total_decrease or 0),
                "first_change": row.first_change,
                "last_change": row.last_change,
                "change_sources": {
                    "rfid": row.rfid_changes or 0,
                    "manual": row.manual_changes or 0,
                    "other": (row.change_count or 0) - 
                            (row.rfid_changes or 0) - 
                            (row.manual_changes or 0)
                }
            }
            for row in results
        ]
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while generating product summary: {str(e)}"
        )

@router.get("/audit/trail")
def get_audit_trail(
    db: Session = Depends(get_db),
    product_sku: str = Query(..., description="Product SKU to audit"),
    start_date: Optional[datetime] = Query(None, description="Start date for audit"),
    end_date: Optional[datetime] = Query(None, description="End date for audit")
):
    """
    Get a complete audit trail for a specific product.
    Shows all inventory changes with before/after quantities.
    """
    try:
        query = db.query(InventoryHistory).options(
            joinedload(InventoryHistory.product)
        ).filter(
            InventoryHistory.product_sku == product_sku.upper()
        ).order_by(
            asc(InventoryHistory.timestamp)
        )
        
        if start_date:
            query = query.filter(InventoryHistory.timestamp >= start_date)
        
        if end_date:
            query = query.filter(InventoryHistory.timestamp <= end_date)
        
        history_entries = query.all()
        
        if not history_entries:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No history found for product {product_sku}"
            )
        
        # Calculate running balance
        audit_trail = []
        current_balance = 0
        
        # Find initial balance if we have a start date
        if start_date:
            # Get the balance before start_date
            last_entry_before = db.query(InventoryHistory).filter(
                InventoryHistory.product_sku == product_sku.upper(),
                InventoryHistory.timestamp < start_date
            ).order_by(
                desc(InventoryHistory.timestamp)
            ).first()
            
            if last_entry_before:
                current_balance = last_entry_before.quantity_after
            else:
                # Check product's current quantity minus changes in our period
                product = db.query(Product).filter(
                    Product.sku == product_sku.upper()
                ).first()
                if product:
                    current_balance = product.quantity
                    # Subtract all changes in our period to get starting balance
                    for entry in history_entries:
                        current_balance -= entry.delta
        
        for entry in history_entries:
            audit_trail.append({
                "timestamp": entry.timestamp,
                "change_type": entry.change_type,
                "source": entry.source,
                "delta": entry.delta,
                "quantity_before": entry.quantity_before,
                "quantity_after": entry.quantity_after,
                "reason": entry.reason,
                "performed_by": entry.performed_by,
                "calculated_balance_before": current_balance,
                "calculated_balance_after": current_balance + entry.delta
            })
            current_balance += entry.delta
        
        return {
            "product_sku": product_sku,
            "product_name": history_entries[0].product.name if history_entries[0].product else None,
            "audit_trail": audit_trail,
            "summary": {
                "total_changes": len(audit_trail),
                "total_increase": sum(e["delta"] for e in audit_trail if e["delta"] > 0),
                "total_decrease": sum(-e["delta"] for e in audit_trail if e["delta"] < 0),
                "net_change": sum(e["delta"] for e in audit_trail),
                "start_balance": audit_trail[0]["calculated_balance_before"] if audit_trail else 0,
                "end_balance": audit_trail[-1]["calculated_balance_after"] if audit_trail else 0
            }
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while generating audit trail: {str(e)}"
        )

# ----------------------------
# Export & Bulk Operations
# ----------------------------
@router.get("/export/csv")
def export_history_csv(
    db: Session = Depends(get_db),
    start_date: Optional[datetime] = Query(None, description="Start date for export"),
    end_date: Optional[datetime] = Query(None, description="End date for export"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU")
):
    """
    Export inventory history as CSV.
    Returns a CSV-formatted string.
    """
    try:
        query = db.query(InventoryHistory).options(
            joinedload(InventoryHistory.product)
        ).order_by(
            desc(InventoryHistory.timestamp)
        )
        
        if start_date:
            query = query.filter(InventoryHistory.timestamp >= start_date)
        
        if end_date:
            query = query.filter(InventoryHistory.timestamp <= end_date)
        
        if product_sku:
            query = query.filter(InventoryHistory.product_sku == product_sku.upper())
        
        history_entries = query.limit(10000).all()  # Limit for safety
        
        # Generate CSV
        csv_lines = [
            "Timestamp,Product SKU,Product Name,Change Type,Delta,Quantity Before,Quantity After,Source,Reason,Performed By"
        ]
        
        for entry in history_entries:
            csv_lines.append(
                f'{entry.timestamp.isoformat() if entry.timestamp else ""},'
                f'{entry.product_sku or ""},'
                f'"{entry.product.name if entry.product else ""}",'
                f'{entry.change_type or ""},'
                f'{entry.delta or 0},'
                f'{entry.quantity_before or 0},'
                f'{entry.quantity_after or 0},'
                f'{entry.source or ""},'
                f'"{entry.reason or ""}",'
                f'{entry.performed_by or ""}'
            )
        
        return {
            "filename": f"inventory_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            "content": "\n".join(csv_lines),
            "count": len(history_entries)
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while exporting history: {str(e)}"
        )

@router.post("/cleanup/old")
def cleanup_old_history(
    db: Session = Depends(get_db),
    older_than_days: int = Query(365, ge=30, le=3650, description="Delete entries older than X days"),
    dry_run: bool = Query(True, description="If true, only show what would be deleted")
):
    """
    Clean up old history entries.
    By default, runs as a dry run (no actual deletion).
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        # Count entries that would be deleted
        count_query = db.query(func.count(InventoryHistory.id)).filter(
            InventoryHistory.timestamp < cutoff_date
        )
        
        entry_count = count_query.scalar() or 0
        
        if dry_run:
            return {
                "dry_run": True,
                "cutoff_date": cutoff_date,
                "entries_to_delete": entry_count,
                "message": f"{entry_count} entries would be deleted (older than {cutoff_date.date()})."
            }
        else:
            # Actually delete the entries
            delete_query = db.query(InventoryHistory).filter(
                InventoryHistory.timestamp < cutoff_date
            )
            
            deleted_count = delete_query.delete(synchronize_session=False)
            db.commit()
            
            return {
                "dry_run": False,
                "cutoff_date": cutoff_date,
                "entries_deleted": deleted_count,
                "message": f"Successfully deleted {deleted_count} history entries older than {cutoff_date.date()}."
            }
        
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during cleanup: {str(e)}"
        )

# ----------------------------
# Statistics
# ----------------------------
@router.get("/stats/summary")
def get_history_stats(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get statistics about inventory history.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic statistics
        stats = db.query(
            func.count(InventoryHistory.id).label("total_entries"),
            func.count(func.distinct(InventoryHistory.product_sku)).label("unique_products"),
            func.sum(
                case(
                    (InventoryHistory.delta > 0, InventoryHistory.delta),
                    else_=0
                )
            ).label("total_increase"),
            func.sum(
                case(
                    (InventoryHistory.delta < 0, -InventoryHistory.delta),
                    else_=0
                )
            ).label("total_decrease"),
            func.avg(InventoryHistory.delta).label("avg_delta"),
            func.min(InventoryHistory.timestamp).label("oldest_entry"),
            func.max(InventoryHistory.timestamp).label("newest_entry")
        ).filter(
            InventoryHistory.timestamp >= cutoff_date
        ).first()
        
        # Breakdown by change type
        change_type_stats = db.query(
            InventoryHistory.change_type,
            func.count(InventoryHistory.id).label("count"),
            func.sum(InventoryHistory.delta).label("total_delta")
        ).filter(
            InventoryHistory.timestamp >= cutoff_date
        ).group_by(
            InventoryHistory.change_type
        ).all()
        
        # Breakdown by source
        source_stats = db.query(
            InventoryHistory.source,
            func.count(InventoryHistory.id).label("count")
        ).filter(
            InventoryHistory.timestamp >= cutoff_date
        ).group_by(
            InventoryHistory.source
        ).all()
        
        # Most active products
        active_products = db.query(
            InventoryHistory.product_sku,
            Product.name,
            func.count(InventoryHistory.id).label("change_count"),
            func.sum(InventoryHistory.delta).label("net_change")
        ).outerjoin(
            Product, InventoryHistory.product_sku == Product.sku
        ).filter(
            InventoryHistory.timestamp >= cutoff_date
        ).group_by(
            InventoryHistory.product_sku,
            Product.name
        ).order_by(
            desc("change_count")
        ).limit(10).all()
        
        return {
            "period_days": days,
            "cutoff_date": cutoff_date,
            "total_entries": stats.total_entries or 0,
            "unique_products": stats.unique_products or 0,
            "total_increase": stats.total_increase or 0,
            "total_decrease": stats.total_decrease or 0,
            "net_change": (stats.total_increase or 0) - (stats.total_decrease or 0),
            "avg_delta": float(stats.avg_delta) if stats.avg_delta else 0,
            "oldest_entry": stats.oldest_entry,
            "newest_entry": stats.newest_entry,
            "breakdown_by_change_type": [
                {
                    "change_type": row.change_type,
                    "count": row.count,
                    "total_delta": row.total_delta
                }
                for row in change_type_stats
            ],
            "breakdown_by_source": [
                {
                    "source": row.source,
                    "count": row.count
                }
                for row in source_stats
            ],
            "most_active_products": [
                {
                    "sku": row.product_sku,
                    "name": row.name,
                    "change_count": row.change_count,
                    "net_change": row.net_change or 0
                }
                for row in active_products
            ]
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching history stats: {str(e)}"
        )