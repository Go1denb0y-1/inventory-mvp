from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import func, and_, or_, case, desc, asc, text

from app.database import get_db, get_db_context
from app.models import (
    Product, Transaction, InventoryHistory,
    TransactionType, ChangeType, SourceType
)
from app.schemas import (
    TransactionCreate, TransactionUpdate, TransactionOut,
    BulkTransactionCreate, PaginationParams, SearchQuery,
    SuccessResponse, ErrorResponse, BulkOperationResult
)

router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"]
)

# ----------------------------
# Helper Functions
# ----------------------------
async def process_transaction_async(
    product_sku: str,
    transaction_type: TransactionType,
    quantity: int,
    db: Session,
    unit_price: Optional[Decimal] = None,
    source: SourceType = SourceType.MANUAL,
    reference: Optional[str] = None,
    note: Optional[str] = None,
    user_id: Optional[str] = None,
    device_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async function to process a transaction with proper transaction handling.
    """
    with get_db_context() as session:
        try:
            # Use SELECT FOR UPDATE to prevent race conditions
            product = (
                session.query(Product)
                .filter(
                    and_(
                        Product.sku == product_sku.upper(),
                        Product.is_active == True
                    )
                )
                .with_for_update()
                .first()
            )
            
            if not product:
                raise ValueError(f"Active product with SKU {product_sku} not found")
            
            before_quantity = product.quantity
            
            # Process the transaction
            if transaction_type == TransactionType.IN:
                delta = quantity
                new_quantity = before_quantity + delta
                change_type = ChangeType.MANUAL_IN
                
            elif transaction_type == TransactionType.OUT:
                if before_quantity < quantity:
                    raise ValueError(
                        f"Insufficient stock. Available: {before_quantity}, Requested: {quantity}"
                    )
                delta = -quantity
                new_quantity = before_quantity + delta
                change_type = ChangeType.MANUAL_OUT
                
            else:
                raise ValueError(f"Invalid transaction type: {transaction_type}")
            
            # Update product quantity
            product.quantity = new_quantity
            product.updated_at = datetime.utcnow()
            
            # Update low stock flag
            if product.min_quantity is not None:
                product.is_low_stock = product.quantity <= product.min_quantity
            
            # Calculate total value if price is available
            total_value = None
            effective_unit_price = unit_price or product.price
            
            if effective_unit_price:
                total_value = effective_unit_price * quantity
            
            # Create transaction record
            transaction = Transaction(
                product_id=product.id,
                product_sku=product.sku,
                transaction_type=transaction_type,
                quantity=quantity,
                unit_price=effective_unit_price,
                total_value=total_value,
                source=source,
                reference=reference,
                note=note,
                user_id=user_id,
                device_id=device_id
            )
            session.add(transaction)
            
            # Create inventory history record
            history = InventoryHistory(
                product_id=product.id,
                product_sku=product.sku,
                change_type=change_type,
                quantity_before=before_quantity,
                quantity_after=new_quantity,
                source=source,
                reason=note or f"Manual transaction: {transaction_type.value}",
                performed_by=user_id or "SYSTEM"
            )
            session.add(history)
            
            # Commit the transaction
            session.commit()
            
            return {
                "success": True,
                "product_sku": product.sku,
                "product_name": product.name,
                "transaction_type": transaction_type.value,
                "quantity": quantity,
                "before_quantity": before_quantity,
                "after_quantity": new_quantity,
                "delta": delta,
                "transaction_id": transaction.id,
                "history_id": history.id,
                "unit_price": effective_unit_price,
                "total_value": total_value
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

# ----------------------------
# CRUD Operations
# ----------------------------
@router.post("/", response_model=TransactionOut, status_code=status.HTTP_201_CREATED)
def create_transaction(
    payload: TransactionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new transaction (incoming or outgoing).
    
    This will update the product quantity and create inventory history.
    For high-throughput scenarios, use the async version.
    """
    try:
        # Validate product exists and is active
        product = db.query(Product).filter(
            and_(
                Product.sku == payload.product_sku.upper(),
                Product.is_active == True
            )
        ).first()
        
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Active product with SKU {payload.product_sku} not found"
            )
        
        # Validate quantity
        if payload.quantity <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quantity must be greater than 0"
            )
        
        # Validate outgoing transaction has sufficient stock
        if payload.transaction_type == TransactionType.OUT:
            if product.quantity < payload.quantity:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Insufficient stock. Available: {product.quantity}, Requested: {payload.quantity}"
                )
        
        # Calculate total value
        total_value = None
        effective_unit_price = payload.unit_price or product.price
        if effective_unit_price:
            total_value = effective_unit_price * payload.quantity
        
        # Use database transaction for consistency
        with get_db_context() as session:
            # Lock the product for update
            product_lock = (
                session.query(Product)
                .filter(Product.id == product.id)
                .with_for_update()
                .first()
            )
            
            before_quantity = product_lock.quantity
            delta = payload.quantity if payload.transaction_type == TransactionType.IN else -payload.quantity
            new_quantity = before_quantity + delta
            
            # Update product
            product_lock.quantity = new_quantity
            product_lock.updated_at = datetime.utcnow()
            
            if product_lock.min_quantity is not None:
                product_lock.is_low_stock = product_lock.quantity <= product_lock.min_quantity
            
            # Create transaction
            transaction = Transaction(
                product_id=product_lock.id,
                product_sku=product_lock.sku,
                transaction_type=payload.transaction_type,
                quantity=payload.quantity,
                unit_price=effective_unit_price,
                total_value=total_value,
                source=payload.source,
                reference=payload.reference,
                note=payload.note,
                user_id=payload.user_id,
                device_id=payload.device_id
            )
            session.add(transaction)
            
            # Create inventory history
            change_type = (
                ChangeType.MANUAL_IN if payload.transaction_type == TransactionType.IN 
                else ChangeType.MANUAL_OUT
            )
            
            history = InventoryHistory(
                product_id=product_lock.id,
                product_sku=product_lock.sku,
                change_type=change_type,
                quantity_before=before_quantity,
                quantity_after=new_quantity,
                source=payload.source,
                reason=payload.note or f"Manual transaction: {payload.transaction_type.value}",
                performed_by=payload.user_id or "SYSTEM"
            )
            session.add(history)
            
            # Commit everything
            session.commit()
            session.refresh(transaction)
            session.refresh(history)
            
            # Add product name to transaction response
            transaction.product_name = product_lock.name
            
            return transaction
            
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while creating transaction: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post("/bulk", response_model=BulkOperationResult, status_code=status.HTTP_202_ACCEPTED)
async def create_bulk_transactions(
    payload: BulkTransactionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create multiple transactions in bulk.
    
    Each transaction is processed independently. Failures in one don't affect others.
    Returns immediately, processing happens in background.
    """
    try:
        if not payload.transactions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No transactions provided"
            )
        
        if len(payload.transactions) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 500 transactions allowed per request"
            )
        
        # Process each transaction in background
        successful = 0
        errors = []
        
        for i, transaction_data in enumerate(payload.transactions):
            try:
                background_tasks.add_task(
                    process_transaction_async,
                    product_sku=transaction_data.product_sku,
                    transaction_type=transaction_data.transaction_type,
                    quantity=transaction_data.quantity,
                    unit_price=transaction_data.unit_price,
                    source=transaction_data.source,
                    reference=(
                        transaction_data.reference or 
                        f"{payload.batch_reference}-{i}" if payload.batch_reference else None
                    ),
                    note=transaction_data.note,
                    user_id=transaction_data.user_id,
                    device_id=transaction_data.device_id,
                    db=db
                )
                successful += 1
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "product_sku": transaction_data.product_sku,
                    "error": str(e)
                })
        
        return BulkOperationResult(
            total=len(payload.transactions),
            successful=successful,
            failed=len(errors),
            errors=errors
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing bulk transactions: {str(e)}"
        )

@router.get("/", response_model=List[TransactionOut])
def list_transactions(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU"),
    transaction_type: Optional[TransactionType] = Query(None, description="Filter by transaction type"),
    source: Optional[SourceType] = Query(None, description="Filter by source"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    reference: Optional[str] = Query(None, description="Filter by reference"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    device_id: Optional[str] = Query(None, description="Filter by device ID"),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'")
):
    """
    List transactions with advanced filtering and pagination.
    """
    try:
        # Build query with product relationship
        query = db.query(Transaction).options(
            joinedload(Transaction.product)
        )
        
        # Apply filters
        filters = []
        
        if product_sku:
            filters.append(Transaction.product_sku == product_sku.upper())
        
        if transaction_type:
            filters.append(Transaction.transaction_type == transaction_type)
        
        if source:
            filters.append(Transaction.source == source)
        
        if start_date:
            filters.append(Transaction.timestamp >= start_date)
        
        if end_date:
            filters.append(Transaction.timestamp <= end_date)
        
        if reference:
            filters.append(Transaction.reference.ilike(f"%{reference}%"))
        
        if user_id:
            filters.append(Transaction.user_id.ilike(f"%{user_id}%"))
        
        if device_id:
            filters.append(Transaction.device_id == device_id)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Apply sorting
        if hasattr(Transaction, sort_by):
            order_func = desc if sort_order.lower() == "desc" else asc
            query = query.order_by(order_func(getattr(Transaction, sort_by)))
        else:
            # Default sorting by timestamp descending
            query = query.order_by(desc(Transaction.timestamp))
        
        # Apply pagination
        total_count = query.count()
        transactions = query.offset(skip).limit(limit).all()
        
        # Enrich with product names
        for transaction in transactions:
            if transaction.product and not hasattr(transaction, 'product_name'):
                transaction.product_name = transaction.product.name
        
        return transactions
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching transactions: {str(e)}"
        )

@router.get("/search", response_model=List[TransactionOut])
def search_transactions(
    db: Session = Depends(get_db),
    q: str = Query(..., min_length=1, description="Search query"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """
    Search transactions by product SKU, name, reference, note, or user ID.
    """
    try:
        query = db.query(Transaction).options(
            joinedload(Transaction.product)
        ).filter(
            or_(
                Transaction.product_sku.ilike(f"%{q}%"),
                Transaction.reference.ilike(f"%{q}%"),
                Transaction.note.ilike(f"%{q}%"),
                Transaction.user_id.ilike(f"%{q}%")
            )
        ).order_by(
            desc(Transaction.timestamp)
        )
        
        total_count = query.count()
        results = query.offset(skip).limit(limit).all()
        
        # Also search by product name if we have products
        if len(results) < limit:
            product_name_query = db.query(Transaction).options(
                joinedload(Transaction.product)
            ).join(
                Product, Transaction.product_sku == Product.sku
            ).filter(
                Product.name.ilike(f"%{q}%")
            ).order_by(
                desc(Transaction.timestamp)
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
            detail=f"Database error while searching transactions: {str(e)}"
        )

@router.get("/{transaction_id}", response_model=TransactionOut)
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific transaction by ID.
    """
    try:
        transaction = db.query(Transaction).options(
            joinedload(Transaction.product)
        ).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction with ID {transaction_id} not found"
            )
        
        # Add product name if available
        if transaction.product and not hasattr(transaction, 'product_name'):
            transaction.product_name = transaction.product.name
        
        return transaction
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching transaction: {str(e)}"
        )

@router.put("/{transaction_id}", response_model=TransactionOut)
def update_transaction(
    transaction_id: int,
    payload: TransactionUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a transaction (non-destructive fields only).
    
    Only allows updating note and reference fields.
    Does not affect product quantities or inventory history.
    """
    try:
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction with ID {transaction_id} not found"
            )
        
        # Only allow updating non-destructive fields
        update_data = payload.dict(exclude_unset=True)
        
        if "note" in update_data:
            transaction.note = update_data["note"]
        
        if "reference" in update_data:
            transaction.reference = update_data["reference"]
        
        transaction.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(transaction)
        
        # Load product for response
        if transaction.product_sku:
            product = db.query(Product).filter(
                Product.sku == transaction.product_sku
            ).first()
            if product:
                transaction.product_name = product.name
        
        return transaction
        
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while updating transaction: {str(e)}"
        )

@router.delete("/{transaction_id}", response_model=SuccessResponse)
def delete_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    force: bool = Query(False, description="Force delete (with inventory correction)")
):
    """
    Delete a transaction.
    
    WARNING: By default, only deletes the transaction record.
    Use force=true to also reverse the inventory impact.
    """
    try:
        transaction = db.query(Transaction).filter(
            Transaction.id == transaction_id
        ).first()
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction with ID {transaction_id} not found"
            )
        
        if force:
            # Reverse the inventory impact
            product = db.query(Product).filter(
                Product.sku == transaction.product_sku
            ).with_for_update().first()
            
            if product:
                if transaction.transaction_type == TransactionType.IN:
                    # Reverse IN transaction: subtract quantity
                    if product.quantity < transaction.quantity:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Cannot reverse transaction: insufficient current stock"
                        )
                    product.quantity -= transaction.quantity
                else:
                    # Reverse OUT transaction: add quantity
                    product.quantity += transaction.quantity
                
                product.updated_at = datetime.utcnow()
                
                # Update low stock flag
                if product.min_quantity is not None:
                    product.is_low_stock = product.quantity <= product.min_quantity
                
                # Create correction history entry
                history = InventoryHistory(
                    product_id=product.id,
                    product_sku=product.sku,
                    change_type=ChangeType.CORRECTION,
                    quantity_before=product.quantity,
                    quantity_after=product.quantity,
                    source=SourceType.SYSTEM,
                    reason=f"Transaction #{transaction_id} deletion correction",
                    performed_by="SYSTEM"
                )
                db.add(history)
        
        # Delete the transaction
        db.delete(transaction)
        db.commit()
        
        return SuccessResponse(
            message=f"Transaction {transaction_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while deleting transaction: {str(e)}"
        )

# ----------------------------
# Analytics & Reports
# ----------------------------
@router.get("/summary/daily")
def get_daily_transaction_summary(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU")
):
    """
    Get daily transaction summary.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            func.date(Transaction.timestamp).label("date"),
            func.count(Transaction.id).label("total_transactions"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("total_incoming"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("total_outgoing"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("total_incoming_value"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("total_outgoing_value")
        ).filter(
            Transaction.timestamp >= cutoff_date
        )
        
        if product_sku:
            query = query.filter(Transaction.product_sku == product_sku.upper())
        
        results = query.group_by(
            func.date(Transaction.timestamp)
        ).order_by(
            func.date(Transaction.timestamp).desc()
        ).all()
        
        return [
            {
                "date": row.date,
                "total_transactions": row.total_transactions or 0,
                "total_incoming": row.total_incoming or 0,
                "total_outgoing": row.total_outgoing or 0,
                "net_movement": (row.total_incoming or 0) - (row.total_outgoing or 0),
                "total_incoming_value": row.total_incoming_value or Decimal('0'),
                "total_outgoing_value": row.total_outgoing_value or Decimal('0'),
                "net_value": (row.total_incoming_value or Decimal('0')) - 
                            (row.total_outgoing_value or Decimal('0'))
            }
            for row in results
        ]
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while generating daily summary: {str(e)}"
        )

@router.get("/summary/by-product")
def get_transaction_summary_by_product(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    limit: int = Query(20, ge=1, le=100, description="Number of products to return")
):
    """
    Get transaction summary grouped by product.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        results = db.query(
            Transaction.product_sku,
            Product.name,
            func.count(Transaction.id).label("transaction_count"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("total_incoming"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("total_outgoing"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("incoming_value"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("outgoing_value"),
            func.avg(Transaction.unit_price).label("avg_unit_price")
        ).outerjoin(
            Product, Transaction.product_sku == Product.sku
        ).filter(
            Transaction.timestamp >= cutoff_date
        ).group_by(
            Transaction.product_sku,
            Product.name
        ).order_by(
            desc("transaction_count")
        ).limit(limit).all()
        
        return [
            {
                "sku": row.product_sku,
                "name": row.name,
                "transaction_count": row.transaction_count or 0,
                "total_incoming": row.total_incoming or 0,
                "total_outgoing": row.total_outgoing or 0,
                "net_movement": (row.total_incoming or 0) - (row.total_outgoing or 0),
                "incoming_value": row.incoming_value or Decimal('0'),
                "outgoing_value": row.outgoing_value or Decimal('0'),
                "net_value": (row.incoming_value or Decimal('0')) - (row.outgoing_value or Decimal('0')),
                "avg_unit_price": row.avg_unit_price or Decimal('0')
            }
            for row in results
        ]
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while generating product summary: {str(e)}"
        )

# ----------------------------
# Export Operations
# ----------------------------
@router.get("/export/csv")
def export_transactions_csv(
    db: Session = Depends(get_db),
    start_date: Optional[datetime] = Query(None, description="Start date for export"),
    end_date: Optional[datetime] = Query(None, description="End date for export"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU"),
    transaction_type: Optional[TransactionType] = Query(None, description="Filter by transaction type")
):
    """
    Export transactions as CSV.
    """
    try:
        query = db.query(Transaction).options(
            joinedload(Transaction.product)
        ).order_by(
            desc(Transaction.timestamp)
        )
        
        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)
        
        if end_date:
            query = query.filter(Transaction.timestamp <= end_date)
        
        if product_sku:
            query = query.filter(Transaction.product_sku == product_sku.upper())
        
        if transaction_type:
            query = query.filter(Transaction.transaction_type == transaction_type)
        
        transactions = query.limit(10000).all()  # Limit for safety
        
        # Generate CSV
        csv_lines = [
            "ID,Timestamp,Product SKU,Product Name,Type,Quantity,Unit Price,Total Value,Source,Reference,Note,User ID,Device ID"
        ]
        
        for transaction in transactions:
            csv_lines.append(
                f'{transaction.id},'
                f'{transaction.timestamp.isoformat() if transaction.timestamp else ""},'
                f'{transaction.product_sku or ""},'
                f'"{transaction.product.name if transaction.product else ""}",'
                f'{transaction.transaction_type or ""},'
                f'{transaction.quantity or 0},'
                f'{transaction.unit_price or 0},'
                f'{transaction.total_value or 0},'
                f'{transaction.source or ""},'
                f'{transaction.reference or ""},'
                f'"{transaction.note or ""}",'
                f'{transaction.user_id or ""},'
                f'{transaction.device_id or ""}'
            )
        
        return {
            "filename": f"transactions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            "content": "\n".join(csv_lines),
            "count": len(transactions)
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while exporting transactions: {str(e)}"
        )

# ----------------------------
# Statistics
# ----------------------------
@router.get("/stats/summary")
def get_transaction_stats(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get transaction statistics.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic statistics
        stats = db.query(
            func.count(Transaction.id).label("total_transactions"),
            func.count(func.distinct(Transaction.product_sku)).label("unique_products"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("total_incoming"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("total_outgoing"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("total_incoming_value"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("total_outgoing_value"),
            func.avg(Transaction.quantity).label("avg_quantity"),
            func.min(Transaction.timestamp).label("oldest_transaction"),
            func.max(Transaction.timestamp).label("newest_transaction")
        ).filter(
            Transaction.timestamp >= cutoff_date
        ).first()
        
        # Breakdown by transaction type
        type_stats = db.query(
            Transaction.transaction_type,
            func.count(Transaction.id).label("count"),
            func.sum(Transaction.quantity).label("total_quantity"),
            func.sum(Transaction.total_value).label("total_value")
        ).filter(
            Transaction.timestamp >= cutoff_date
        ).group_by(
            Transaction.transaction_type
        ).all()
        
        # Breakdown by source
        source_stats = db.query(
            Transaction.source,
            func.count(Transaction.id).label("count"),
            func.sum(Transaction.quantity).label("total_quantity")
        ).filter(
            Transaction.timestamp >= cutoff_date
        ).group_by(
            Transaction.source
        ).all()
        
        # Most active products
        active_products = db.query(
            Transaction.product_sku,
            Product.name,
            func.count(Transaction.id).label("transaction_count"),
            func.sum(Transaction.quantity).label("total_quantity")
        ).outerjoin(
            Product, Transaction.product_sku == Product.sku
        ).filter(
            Transaction.timestamp >= cutoff_date
        ).group_by(
            Transaction.product_sku,
            Product.name
        ).order_by(
            desc("transaction_count")
        ).limit(10).all()
        
        return {
            "period_days": days,
            "cutoff_date": cutoff_date,
            "total_transactions": stats.total_transactions or 0,
            "unique_products": stats.unique_products or 0,
            "total_incoming": stats.total_incoming or 0,
            "total_outgoing": stats.total_outgoing or 0,
            "net_movement": (stats.total_incoming or 0) - (stats.total_outgoing or 0),
            "total_incoming_value": stats.total_incoming_value or Decimal('0'),
            "total_outgoing_value": stats.total_outgoing_value or Decimal('0'),
            "net_value": (stats.total_incoming_value or Decimal('0')) - 
                        (stats.total_outgoing_value or Decimal('0')),
            "avg_quantity": float(stats.avg_quantity) if stats.avg_quantity else 0,
            "oldest_transaction": stats.oldest_transaction,
            "newest_transaction": stats.newest_transaction,
            "breakdown_by_type": [
                {
                    "transaction_type": row.transaction_type,
                    "count": row.count,
                    "total_quantity": row.total_quantity or 0,
                    "total_value": row.total_value or Decimal('0')
                }
                for row in type_stats
            ],
            "breakdown_by_source": [
                {
                    "source": row.source,
                    "count": row.count,
                    "total_quantity": row.total_quantity or 0
                }
                for row in source_stats
            ],
            "most_active_products": [
                {
                    "sku": row.product_sku,
                    "name": row.name,
                    "transaction_count": row.transaction_count,
                    "total_quantity": row.total_quantity or 0
                }
                for row in active_products
            ]
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while fetching transaction stats: {str(e)}"
        )