from typing import Dict, List, Optional, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
from collections import defaultdict

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import func, case, and_, or_, text, extract
from sqlalchemy.orm import Session, aliased
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.models import (
    Product, Transaction, InventoryHistory, 
    TransactionType, ChangeType, SourceType
)
from app.schemas import (
    DashboardStats, SalesReport, StockLevelReport, 
    TransactionReport, PaginationParams, SearchQuery
)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)

# ----------------------------
# Dashboard Statistics
# ----------------------------
@router.get("/dashboard", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    include_inactive: bool = Query(False, description="Include inactive products")
):
    """
    Get comprehensive dashboard statistics.
    """
    try:
        today = date.today()
        week_ago = today - timedelta(days=7)
        
        # Base query filter
        product_filter = []
        if not include_inactive:
            product_filter.append(Product.is_active == True)
        
        # 1. Product Statistics
        product_stats = db.query(
            func.count(Product.id).label("total_products"),
            func.sum(
                case((Product.is_active == True, 1), else_=0)
            ).label("active_products"),
            func.sum(Product.quantity).label("total_items_in_stock"),
            func.sum(
                case(
                    (Product.quantity <= Product.min_quantity, 1),
                    else_=0
                )
            ).label("low_stock_items"),
            func.sum(
                case((Product.quantity == 0, 1), else_=0)
            ).label("out_of_stock_items"),
            func.sum(
                case(
                    (Product.quantity <= Product.reorder_point, 1),
                    else_=0
                )
            ).label("items_needing_reorder"),
            func.sum(Product.quantity * Product.price).label("total_inventory_value")
        ).filter(*product_filter).first()
        
        # 2. Transaction Statistics
        transaction_stats = db.query(
            func.count(Transaction.id).label("transactions_today"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("incoming_today"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("outgoing_today"),
        ).filter(
            func.date(Transaction.timestamp) == today
        ).first()
        
        # 3. Weekly Transactions
        weekly_transactions = db.query(
            func.count(Transaction.id).label("transactions_this_week")
        ).filter(
            func.date(Transaction.timestamp) >= week_ago
        ).scalar() or 0
        
        # 4. RFID Statistics
        rfid_stats = db.query(
            func.count(Product.id).label("total_rfid_tags"),
            func.count(
                case(
                    (and_(
                        Product.rfid_tag.isnot(None),
                        Product.is_active == True
                    ), 1)
                )
            ).label("active_rfid_tags")
        ).filter(*product_filter).first()
        
        # Calculate net movement
        net_movement = (transaction_stats.incoming_today or 0) - (transaction_stats.outgoing_today or 0)
        
        return DashboardStats(
            total_products=product_stats.total_products or 0,
            active_products=product_stats.active_products or 0,
            total_items_in_stock=product_stats.total_items_in_stock or 0,
            total_inventory_value=product_stats.total_inventory_value or Decimal('0'),
            low_stock_items=product_stats.low_stock_items or 0,
            out_of_stock_items=product_stats.out_of_stock_items or 0,
            transactions_today=transaction_stats.transactions_today or 0,
            transactions_this_week=weekly_transactions,
            total_rfid_tags=rfid_stats.total_rfid_tags or 0,
            active_rfid_tags=rfid_stats.active_rfid_tags or 0,
            net_movement_today=net_movement
        )
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching dashboard stats: {str(e)}"
        )

# ----------------------------
# Movement Analytics
# ----------------------------
@router.get("/movement/summary")
def get_movement_summary(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU")
):
    """
    Get inventory movement summary (net movement per product).
    Positive = net inflow, Negative = net outflow.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            Transaction.product_sku,
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=-Transaction.quantity
                )
            ).label("net_movement"),
            func.count(Transaction.id).label("transaction_count"),
            func.min(Transaction.timestamp).label("first_transaction"),
            func.max(Transaction.timestamp).label("last_transaction"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("total_in"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("total_out")
        ).filter(
            Transaction.timestamp >= cutoff_date
        )
        
        if product_sku:
            query = query.filter(Transaction.product_sku == product_sku.upper())
        
        results = query.group_by(Transaction.product_sku).all()
        
        return [
            {
                "sku": sku,
                "net_movement": net_movement or 0,
                "transaction_count": transaction_count or 0,
                "total_in": total_in or 0,
                "total_out": total_out or 0,
                "first_transaction": first_transaction.isoformat() if first_transaction else None,
                "last_transaction": last_transaction.isoformat() if last_transaction else None,
                "movement_rate": (net_movement or 0) / days if days > 0 else 0
            }
            for sku, net_movement, transaction_count, first_transaction, last_transaction, total_in, total_out in results
        ]
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching movement summary: {str(e)}"
        )

@router.get("/movement/daily")
def get_daily_movement(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    product_sku: Optional[str] = Query(None, description="Filter by product SKU")
):
    """
    Get daily inventory movement breakdown.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(
            func.date(Transaction.timestamp).label("date"),
            Transaction.product_sku,
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("incoming"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("outgoing"),
            func.count(Transaction.id).label("transaction_count")
        ).filter(
            Transaction.timestamp >= cutoff_date
        )
        
        if product_sku:
            query = query.filter(Transaction.product_sku == product_sku.upper())
        
        results = query.group_by(
            func.date(Transaction.timestamp),
            Transaction.product_sku
        ).order_by(
            func.date(Transaction.timestamp).desc()
        ).all()
        
        # Group by date for easier consumption
        daily_summary = defaultdict(lambda: {
            "date": None,
            "total_incoming": 0,
            "total_outgoing": 0,
            "net_movement": 0,
            "total_transactions": 0,
            "products": []
        })
        
        for date_str, sku, incoming, outgoing, count in results:
            if date_str not in daily_summary:
                daily_summary[date_str] = {
                    "date": date_str,
                    "total_incoming": 0,
                    "total_outgoing": 0,
                    "net_movement": 0,
                    "total_transactions": 0,
                    "products": []
                }
            
            daily_summary[date_str]["total_incoming"] += (incoming or 0)
            daily_summary[date_str]["total_outgoing"] += (outgoing or 0)
            daily_summary[date_str]["total_transactions"] += (count or 0)
            daily_summary[date_str]["net_movement"] += (incoming or 0) - (outgoing or 0)
            
            daily_summary[date_str]["products"].append({
                "sku": sku,
                "incoming": incoming or 0,
                "outgoing": outgoing or 0,
                "net_movement": (incoming or 0) - (outgoing or 0),
                "transaction_count": count or 0
            })
        
        return list(daily_summary.values())
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching daily movement: {str(e)}"
        )

# ----------------------------
# Sales Analytics
# ----------------------------
@router.get("/sales/report", response_model=SalesReport)
def get_sales_report(
    db: Session = Depends(get_db),
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """
    Generate sales report for a given period.
    """
    try:
        if start_date > end_date:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        # Get outgoing transactions (sales)
        query = db.query(
            Transaction.product_sku,
            func.sum(Transaction.quantity).label("total_quantity"),
            func.sum(Transaction.total_value).label("total_sales"),
            func.avg(Transaction.unit_price).label("avg_price"),
            Product.name,
            Product.category,
            Product.cost
        ).join(
            Product, Transaction.product_sku == Product.sku
        ).filter(
            Transaction.transaction_type == TransactionType.OUT,
            func.date(Transaction.timestamp) >= start_date,
            func.date(Transaction.timestamp) <= end_date,
            Product.is_active == True
        )
        
        if category:
            query = query.filter(Product.category == category)
        
        sales_data = query.group_by(
            Transaction.product_sku,
            Product.name,
            Product.category,
            Product.cost
        ).all()
        
        # Calculate totals
        total_sales = sum(item.total_sales or Decimal('0') for item in sales_data)
        total_quantity = sum(item.total_quantity or 0 for item in sales_data)
        
        # Calculate costs and profit
        sales_with_cost = []
        total_cost = Decimal('0')
        
        for item in sales_data:
            item_cost = (item.cost or Decimal('0')) * (item.total_quantity or 0)
            total_cost += item_cost
            profit = (item.total_sales or Decimal('0')) - item_cost
            
            sales_with_cost.append({
                "sku": item.product_sku,
                "name": item.name,
                "category": item.category,
                "quantity": item.total_quantity or 0,
                "sales": item.total_sales or Decimal('0'),
                "cost": item_cost,
                "profit": profit,
                "profit_margin": (profit / item_cost * 100) if item_cost > 0 else 0
            })
        
        # Calculate profit margin
        profit_margin = ((total_sales - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        # Group by category
        sales_by_category = defaultdict(lambda: {"sales": Decimal('0'), "quantity": 0})
        for item in sales_with_cost:
            cat = item["category"] or "Uncategorized"
            sales_by_category[cat]["sales"] += item["sales"]
            sales_by_category[cat]["quantity"] += item["quantity"]
        
        # Get daily breakdown
        daily_sales = db.query(
            func.date(Transaction.timestamp).label("date"),
            func.sum(Transaction.total_value).label("daily_sales"),
            func.count(Transaction.id).label("transaction_count")
        ).filter(
            Transaction.transaction_type == TransactionType.OUT,
            func.date(Transaction.timestamp) >= start_date,
            func.date(Transaction.timestamp) <= end_date
        ).group_by(
            func.date(Transaction.timestamp)
        ).order_by(
            func.date(Transaction.timestamp)
        ).all()
        
        return SalesReport(
            period_start=datetime.combine(start_date, datetime.min.time()),
            period_end=datetime.combine(end_date, datetime.max.time()),
            total_sales=total_sales,
            total_cost=total_cost,
            total_profit=total_sales - total_cost,
            profit_margin=profit_margin,
            top_products=sorted(sales_with_cost, key=lambda x: x["sales"], reverse=True)[:10],
            sales_by_category=[
                {"category": cat, "sales": data["sales"], "quantity": data["quantity"]}
                for cat, data in sales_by_category.items()
            ],
            sales_by_day=[
                {"date": item.date, "sales": item.daily_sales or Decimal('0'), "transactions": item.transaction_count or 0}
                for item in daily_sales
            ]
        )
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while generating sales report: {str(e)}"
        )

# ----------------------------
# Stock Level Analytics
# ----------------------------
@router.get("/stock/levels", response_model=List[StockLevelReport])
def get_stock_level_report(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None, description="Filter by category"),
    low_stock_only: bool = Query(False, description="Only show low stock items"),
    out_of_stock_only: bool = Query(False, description="Only show out of stock items")
):
    """
    Get detailed stock level report.
    """
    try:
        query = db.query(
            Product.category,
            func.count(Product.id).label("total_items"),
            func.sum(Product.quantity).label("total_quantity"),
            func.sum(Product.quantity * Product.price).label("total_value"),
            func.sum(
                case(
                    (Product.quantity <= Product.min_quantity, 1),
                    else_=0
                )
            ).label("low_stock_items"),
            func.sum(
                case((Product.quantity == 0, 1), else_=0)
            ).label("out_of_stock_items"),
            func.sum(
                case(
                    (Product.quantity <= Product.reorder_point, 1),
                    else_=0
                )
            ).label("items_needing_reorder"),
            func.avg(Product.quantity).label("avg_stock_level")
        ).filter(
            Product.is_active == True
        )
        
        if category:
            query = query.filter(Product.category == category)
        
        if low_stock_only:
            query = query.filter(Product.quantity <= Product.min_quantity)
        
        if out_of_stock_only:
            query = query.filter(Product.quantity == 0)
        
        results = query.group_by(Product.category).all()
        
        reports = []
        for row in results:
            reports.append(StockLevelReport(
                category=row.category,
                total_items=row.total_items or 0,
                total_quantity=row.total_quantity or 0,
                total_value=row.total_value or Decimal('0'),
                low_stock_items=row.low_stock_items or 0,
                out_of_stock_items=row.out_of_stock_items or 0,
                items_needing_reorder=row.items_needing_reorder or 0,
                avg_stock_level=row.avg_stock_level or Decimal('0')
            ))
        
        return reports
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching stock levels: {str(e)}"
        )

# ----------------------------
# Transaction Analytics
# ----------------------------
@router.get("/transactions/report", response_model=TransactionReport)
def get_transaction_report(
    db: Session = Depends(get_db),
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    source: Optional[SourceType] = Query(None, description="Filter by source")
):
    """
    Generate transaction report for a given period.
    """
    try:
        if start_date > end_date:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        # Base query
        query = db.query(Transaction).filter(
            func.date(Transaction.timestamp) >= start_date,
            func.date(Transaction.timestamp) <= end_date
        )
        
        if source:
            query = query.filter(Transaction.source == source)
        
        # Get transaction statistics
        stats = query.with_entities(
            func.count(Transaction.id).label("total_transactions"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, 1),
                    else_=0
                )
            ).label("incoming_transactions"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, 1),
                    else_=0
                )
            ).label("outgoing_transactions"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("incoming_quantity"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("outgoing_quantity")
        ).first()
        
        # Get transactions by source
        by_source = db.query(
            Transaction.source,
            func.count(Transaction.id).label("count")
        ).filter(
            func.date(Transaction.timestamp) >= start_date,
            func.date(Transaction.timestamp) <= end_date
        ).group_by(Transaction.source).all()
        
        # Get transactions by user
        by_user = db.query(
            Transaction.user_id,
            func.count(Transaction.id).label("count")
        ).filter(
            Transaction.user_id.isnot(None),
            func.date(Transaction.timestamp) >= start_date,
            func.date(Transaction.timestamp) <= end_date
        ).group_by(Transaction.user_id).all()
        
        # Get daily summary
        daily_summary = db.query(
            func.date(Transaction.timestamp).label("date"),
            func.count(Transaction.id).label("transaction_count"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.IN, Transaction.quantity),
                    else_=0
                )
            ).label("incoming_quantity"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("outgoing_quantity")
        ).filter(
            func.date(Transaction.timestamp) >= start_date,
            func.date(Transaction.timestamp) <= end_date
        ).group_by(
            func.date(Transaction.timestamp)
        ).order_by(
            func.date(Transaction.timestamp)
        ).all()
        
        return TransactionReport(
            period_start=datetime.combine(start_date, datetime.min.time()),
            period_end=datetime.combine(end_date, datetime.max.time()),
            total_transactions=stats.total_transactions or 0,
            incoming_transactions=stats.incoming_transactions or 0,
            outgoing_transactions=stats.outgoing_transactions or 0,
            transactions_by_source={item.source.value: item.count for item in by_source},
            transactions_by_user={item.user_id: item.count for item in by_user},
            daily_summary=[
                {
                    "date": item.date,
                    "transaction_count": item.transaction_count or 0,
                    "incoming_quantity": item.incoming_quantity or 0,
                    "outgoing_quantity": item.outgoing_quantity or 0,
                    "net_movement": (item.incoming_quantity or 0) - (item.outgoing_quantity or 0)
                }
                for item in daily_summary
            ]
        )
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while generating transaction report: {str(e)}"
        )

# ----------------------------
# Product Performance Analytics
# ----------------------------
@router.get("/products/top-performing")
def get_top_performing_products(
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    limit: int = Query(10, ge=1, le=100, description="Number of products to return")
):
    """
    Get top performing products by sales volume.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        results = db.query(
            Product.sku,
            Product.name,
            Product.category,
            Product.quantity,
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("units_sold"),
            func.sum(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.total_value),
                    else_=Decimal('0')
                )
            ).label("revenue"),
            func.avg(Transaction.unit_price).label("avg_selling_price"),
            func.count(Transaction.id).label("transaction_count")
        ).outerjoin(
            Transaction, and_(
                Transaction.product_sku == Product.sku,
                Transaction.timestamp >= cutoff_date
            )
        ).filter(
            Product.is_active == True
        ).group_by(
            Product.sku,
            Product.name,
            Product.category,
            Product.quantity
        ).order_by(
            text("revenue DESC NULLS LAST")
        ).limit(limit).all()
        
        return [
            {
                "sku": sku,
                "name": name,
                "category": category,
                "current_stock": quantity,
                "units_sold": units_sold or 0,
                "revenue": revenue or Decimal('0'),
                "avg_selling_price": avg_selling_price or Decimal('0'),
                "transaction_count": transaction_count or 0,
                "turnover_rate": (units_sold or 0) / max(quantity, 1) if quantity > 0 else 0
            }
            for sku, name, category, quantity, units_sold, revenue, avg_selling_price, transaction_count in results
        ]
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while fetching top products: {str(e)}"
        )

# ----------------------------
# Inventory Health Analytics
# ----------------------------
@router.get("/inventory/health")
def get_inventory_health(
    db: Session = Depends(get_db),
    threshold_days: int = Query(90, ge=1, le=365, description="Days without movement to consider stagnant")
):
    """
    Analyze inventory health metrics.
    """
    try:
        stagnant_cutoff = datetime.utcnow() - timedelta(days=threshold_days)
        
        # Get products with no recent movement
        stagnant_products = db.query(
            Product.sku,
            Product.name,
            Product.quantity,
            Product.price,
            func.max(Transaction.timestamp).label("last_transaction")
        ).outerjoin(
            Transaction, Transaction.product_sku == Product.sku
        ).filter(
            Product.is_active == True,
            Product.quantity > 0
        ).group_by(
            Product.sku,
            Product.name,
            Product.quantity,
            Product.price
        ).having(
            func.max(Transaction.timestamp) < stagnant_cutoff
        ).all()
        
        # Calculate stagnant inventory value
        stagnant_value = sum(
            (p.quantity or 0) * (p.price or Decimal('0'))
            for p in stagnant_products
        )
        
        # Get aging inventory
        aging_inventory = db.query(
            Product.sku,
            Product.name,
            Product.quantity,
            Product.price,
            func.min(Transaction.timestamp).label("first_transaction"),
            func.max(Transaction.timestamp).label("last_transaction"),
            (datetime.utcnow() - func.max(Transaction.timestamp)).label("days_since_last_move")
        ).outerjoin(
            Transaction, Transaction.product_sku == Product.sku
        ).filter(
            Product.is_active == True,
            Product.quantity > 0
        ).group_by(
            Product.sku,
            Product.name,
            Product.quantity,
            Product.price
        ).all()
        
        # Calculate inventory turnover
        total_inventory_value = db.query(
            func.sum(Product.quantity * Product.price)
        ).filter(
            Product.is_active == True
        ).scalar() or Decimal('0')
        
        total_sales_last_year = db.query(
            func.sum(Transaction.total_value)
        ).filter(
            Transaction.transaction_type == TransactionType.OUT,
            Transaction.timestamp >= datetime.utcnow() - timedelta(days=365)
        ).scalar() or Decimal('0')
        
        turnover_ratio = (total_sales_last_year / total_inventory_value) if total_inventory_value > 0 else 0
        
        return {
            "stagnant_products": len(stagnant_products),
            "stagnant_inventory_value": stagnant_value,
            "stagnant_items": [
                {
                    "sku": p.sku,
                    "name": p.name,
                    "quantity": p.quantity,
                    "value": (p.quantity or 0) * (p.price or Decimal('0')),
                    "last_transaction": p.last_transaction.isoformat() if p.last_transaction else None
                }
                for p in stagnant_products[:20]  # Limit for performance
            ],
            "aging_inventory": [
                {
                    "sku": p.sku,
                    "name": p.name,
                    "quantity": p.quantity,
                    "value": (p.quantity or 0) * (p.price or Decimal('0')),
                    "first_transaction": p.first_transaction.isoformat() if p.first_transaction else None,
                    "last_transaction": p.last_transaction.isoformat() if p.last_transaction else None,
                    "days_since_last_move": p.days_since_last_move.days if p.days_since_last_move else None
                }
                for p in aging_inventory[:50]
            ],
            "turnover_metrics": {
                "total_inventory_value": total_inventory_value,
                "annual_sales": total_sales_last_year,
                "turnover_ratio": turnover_ratio,
                "days_inventory_outstanding": 365 / turnover_ratio if turnover_ratio > 0 else 0
            }
        }
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while analyzing inventory health: {str(e)}"
        )

# ----------------------------
# Predictive Analytics (Basic)
# ----------------------------
@router.get("/predictions/restocking")
def get_restocking_predictions(
    db: Session = Depends(get_db),
    horizon_days: int = Query(30, ge=1, le=90, description="Prediction horizon in days")
):
    """
    Get restocking predictions based on historical sales.
    """
    try:
        # Get products with sales history
        products_with_sales = db.query(
            Product.sku,
            Product.name,
            Product.quantity,
            Product.reorder_point,
            Product.min_quantity,
            func.avg(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("avg_daily_sales"),
            func.stddev(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.quantity),
                    else_=0
                )
            ).label("sales_stddev"),
            func.count(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.id),
                    else_=None
                )
            ).label("sales_count")
        ).outerjoin(
            Transaction, and_(
                Transaction.product_sku == Product.sku,
                Transaction.timestamp >= datetime.utcnow() - timedelta(days=90)
            )
        ).filter(
            Product.is_active == True,
            Product.reorder_point.isnot(None)
        ).group_by(
            Product.sku,
            Product.name,
            Product.quantity,
            Product.reorder_point,
            Product.min_quantity
        ).having(
            func.count(
                case(
                    (Transaction.transaction_type == TransactionType.OUT, Transaction.id),
                    else_=None
                )
            ) >= 5  # Minimum sales history
        ).all()
        
        predictions = []
        for product in products_with_sales:
            avg_daily_sales = product.avg_daily_sales or 0
            safety_stock = (product.sales_stddev or 0) * 1.65  # 95% confidence
            lead_time_demand = avg_daily_sales * horizon_days + safety_stock
            
            days_until_reorder = max(
                0,
                (product.quantity - (product.reorder_point or 0)) / max(avg_daily_sales, 0.1)
            )
            
            predictions.append({
                "sku": product.sku,
                "name": product.name,
                "current_stock": product.quantity,
                "reorder_point": product.reorder_point,
                "avg_daily_sales": avg_daily_sales,
                "predicted_demand": lead_time_demand,
                "days_until_reorder": int(days_until_reorder),
                "suggested_order_quantity": max(
                    (product.reorder_point or 0) + lead_time_demand - product.quantity,
                    0
                ),
                "urgency": "critical" if days_until_reorder <= 3 else
                          "high" if days_until_reorder <= 7 else
                          "medium" if days_until_reorder <= 14 else "low"
            })
        
        return sorted(predictions, key=lambda x: x["urgency"])
        
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error while generating predictions: {str(e)}"
        )