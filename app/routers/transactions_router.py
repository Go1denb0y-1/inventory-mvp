from typing import List, Optional
from datetime import datetime

from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import and_, or_, desc, asc

from app.database import get_db, get_db_context
from app.models import Product, Transaction, InventoryHistory, ChangeType, SourceType, TransactionType
from app.schemas import (
    TransactionCreate, TransactionUpdate, TransactionOut,
)

router = APIRouter(prefix="/transactions", tags=["Transactions"])

# ----------------------------------------------------------------------
# CRUD Operations
# ----------------------------------------------------------------------

@router.post("/", response_model=TransactionOut, status_code=status.HTTP_201_CREATED)
def create_transaction(
    payload: TransactionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new transaction (incoming or outgoing).

    This will update the product quantity and create inventory history.
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
        query = db.query(Transaction).options(
            joinedload(Transaction.product)
        )

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

        # Sorting
        if hasattr(Transaction, sort_by):
            order_func = desc if sort_order.lower() == "desc" else asc
            query = query.order_by(order_func(getattr(Transaction, sort_by)))
        else:
            query = query.order_by(desc(Transaction.timestamp))

        # Pagination
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

        results = query.offset(skip).limit(limit).all()

        # Also search by product name if we haven't reached the limit
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


@router.delete("/{transaction_id}", response_model=dict)
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

        return {"message": f"Transaction {transaction_id} deleted successfully"}

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while deleting transaction: {str(e)}"
        )