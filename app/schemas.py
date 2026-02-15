from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
from decimal import Decimal
from enum import Enum
import re

# ----------------------------
# Enums
# ----------------------------
class TransactionType(str, Enum):
    IN = "IN"
    OUT = "OUT"

class ChangeType(str, Enum):
    RFID_IN = "RFID_IN"
    RFID_OUT = "RFID_OUT"
    MANUAL_IN = "MANUAL_IN"
    MANUAL_OUT = "MANUAL_OUT"
    ADJUSTMENT = "ADJUSTMENT"
    CORRECTION = "CORRECTION"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"

class SourceType(str, Enum):
    MANUAL = "MANUAL"
    RFID = "RFID"
    API = "API"
    SYSTEM = "SYSTEM"
    IMPORT = "IMPORT"

# ---------------------------------------------------------
# 1. THE SYNC PAYLOAD (EXACTLY AS YOU REQUESTED)
# ---------------------------------------------------------
class ProductPayload(BaseModel):
    """
    This is the strict schema for the Friend API.
    It ONLY contains the fields you explicitly asked for.
    """
    sku: str = Field(..., description="Product SKU (string)")
    name: str
    category: Optional[str] = None
    quantity: int
    rfid_tag: Optional[str] = None
    price: Optional[float] = None  # Float as requested
    cost: Optional[float] = None   # Float as requested
    tags: List[str] = []
    location: Optional[str] = None
    supplier: Optional[str] = None
    is_active: bool = True
    last_updated: Optional[datetime] = None
    source_system: str

    # This allows us to convert a DB object into this payload easily
    model_config = ConfigDict(from_attributes=True)

# ---------------------------------------------------------
# 2. INTERNAL DATABASE SCHEMAS (WITH EXTRA LOGIC)
# ---------------------------------------------------------
class ProductBase(BaseModel):
    sku: str = Field(..., description="Product SKU (string)")
    name: str
    category: Optional[str] = None
    quantity: int
    rfid_tag: Optional[str] = None
    price: Optional[float] = None  # Float as requested
    cost: Optional[float] = None   # Float as requested
    tags: List[str] = []
    location: Optional[str] = None
    supplier: Optional[str] = None
    is_active: bool = True
    last_updated: Optional[datetime] = None
    source_system: str

class ProductCreate(ProductBase):
    sku: str
    
    @field_validator('sku')
    @classmethod
    def validate_sku(cls, v: str) -> str:
        return v.upper().strip()

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    quantity: Optional[int] = Field(None, ge=0)
    price: Optional[Decimal] = Field(None, ge=0)
    cost: Optional[Decimal] = Field(None, ge=0)
    barcode: Optional[str] = None
    location: Optional[str] = None
    supplier: Optional[str] = None
    min_quantity: Optional[int] = Field(None, ge=0)
    max_quantity: Optional[int] = Field(None, ge=0)
    reorder_point: Optional[int] = Field(None, ge=0)
    is_active: bool = True
    tags: Optional[List[str]] = None
    rfid_tag: Optional[str] = None
    source_system: Optional[str] = None # Allow updating source

class ProductOut(ProductBase):
    sku: str
    

    model_config = ConfigDict(from_attributes=True)

# ... (Keep the rest of your RFID, Transaction, and History schemas as they were)
# ----------------------------
# RFID Schemas
# ----------------------------
class RFIDScan(BaseModel):
    """Schema for RFID scan operations"""
    rfid_tag: str = Field(..., min_length=1, max_length=100, description="RFID tag UID")
    mode: TransactionType = Field(..., description="Scan mode: IN or OUT")
    quantity: int = Field(1, gt=0, description="Quantity to move")
    source: SourceType = Field(SourceType.RFID, description="Source of the scan")
    device_id: Optional[str] = Field(None, description="Device that performed the scan")
    location: Optional[str] = Field(None, description="Scan location")
    
    @field_validator('rfid_tag')
    @classmethod
    def validate_rfid_tag(cls, v: str) -> str:
        """Validate RFID tag format"""
        v = v.strip().upper()
        if not v:
            raise ValueError("RFID tag cannot be empty")
        if len(v) > 100:
            raise ValueError("RFID tag cannot exceed 100 characters")
        if not re.match(r'^[A-F0-9]+$', v):
            raise ValueError("RFID tag should contain only hexadecimal characters (0-9, A-F)")
        return v

class RFIDTagCreate(BaseModel):
    """Schema for creating/managing RFID tags"""
    tag_uid: str = Field(..., min_length=1, max_length=100, description="RFID tag UID")
    product_sku: str = Field(..., min_length=1, max_length=50, description="Product SKU to associate")
    location: Optional[str] = Field(None, max_length=100, description="Tag location")
    notes: Optional[str] = Field(None, max_length=500, description="Additional notes")

class RFIDTagOut(BaseModel):
    """Schema for RFID tag responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    tag_uid: str
    product_sku: str
    product_name: Optional[str] = None
    is_active: bool
    last_seen: Optional[datetime]
    first_seen: datetime
    location: Optional[str]
    device_id: Optional[str]
    notes: Optional[str]

# ----------------------------
# Transaction Schemas
# ----------------------------
class TransactionCreate(BaseModel):
    """Schema for creating a transaction"""
    product_sku: str = Field(..., min_length=1, max_length=50, description="Product SKU")
    transaction_type: TransactionType = Field(..., description="Type: IN or OUT")
    quantity: int = Field(..., gt=0, description="Quantity to move")
    unit_price: Optional[Decimal] = Field(None, ge=0, max_digits=10, decimal_places=2, description="Price per unit")
    source: SourceType = Field(SourceType.MANUAL, description="Source of transaction")
    reference: Optional[str] = Field(None, max_length=100, description="External reference")
    note: Optional[str] = Field(None, max_length=500, description="Additional notes")
    user_id: Optional[str] = Field(None, max_length=100, description="User who performed the transaction")
    device_id: Optional[str] = Field(None, max_length=100, description="Device used")

class TransactionUpdate(BaseModel):
    """Schema for updating a transaction"""
    note: Optional[str] = Field(None, max_length=500)
    reference: Optional[str] = Field(None, max_length=100)

class TransactionOut(BaseModel):
    """Schema for transaction responses"""
    model_config = ConfigDict(from_attributes=True)
    
    
    product_id: int
    product_sku: str
    product_name: Optional[str] = None
    transaction_type: TransactionType
    quantity: int
    unit_price: Optional[Decimal]
    total_value: Optional[Decimal]
    timestamp: datetime
    source: SourceType
    reference: Optional[str]
    note: Optional[str]
    user_id: Optional[str]
    device_id: Optional[str]

class BulkTransactionCreate(BaseModel):
    """Schema for creating multiple transactions at once"""
    transactions: List[TransactionCreate] = Field(..., min_items=1, max_items=100)
    batch_reference: Optional[str] = Field(None, max_length=100, description="Batch reference for all transactions")

# ----------------------------
# Inventory History Schemas
# ----------------------------
class InventoryHistoryCreate(BaseModel):
    """Schema for creating inventory history entries"""
    product_sku: str = Field(..., min_length=1, max_length=50)
    change_type: ChangeType = Field(...)
    quantity_before: int = Field(..., ge=0)
    quantity_after: int = Field(..., ge=0)
    source: Optional[SourceType] = None
    reason: Optional[str] = Field(None, max_length=500)
    performed_by: Optional[str] = Field(None, max_length=100)

class InventoryHistoryOut(BaseModel):
    """Schema for inventory history responses"""
    model_config = ConfigDict(from_attributes=True)
    
    product_id: int
    product_sku: str
    product_name: Optional[str] = None
    change_type: ChangeType
    delta: int
    quantity_before: int
    quantity_after: int
    transaction_id: Optional[int]
    source: Optional[SourceType]
    reason: Optional[str]
    performed_by: Optional[str]
    timestamp: datetime

class InventoryHistoryFilter(BaseModel):
    """Schema for filtering inventory history"""
    product_sku: Optional[str] = None
    change_type: Optional[ChangeType] = None
    source: Optional[SourceType] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)

# ----------------------------
# Dashboard & Analytics Schemas
# ----------------------------
class DashboardStats(BaseModel):
    """Schema for dashboard statistics"""
    total_products: int
    active_products: int
    total_items_in_stock: int
    total_inventory_value: Decimal
    low_stock_items: int
    out_of_stock_items: int
    transactions_today: int
    transactions_this_week: int
    total_rfid_tags: int
    active_rfid_tags: int

class SalesReport(BaseModel):
    """Schema for sales reports"""
    period_start: datetime
    period_end: datetime
    total_sales: Decimal
    total_cost: Decimal
    total_profit: Decimal
    profit_margin: Decimal
    top_products: List[Dict[str, Any]]
    sales_by_category: List[Dict[str, Any]]
    sales_by_day: List[Dict[str, Any]]

class StockLevelReport(BaseModel):
    """Schema for stock level reports"""
    category: Optional[str]
    total_items: int
    total_value: Decimal
    out_of_stock_items: int
    avg_stock_level: Decimal

class TransactionReport(BaseModel):
    """Schema for transaction reports"""
    period_start: datetime
    period_end: datetime
    total_transactions: int
    incoming_transactions: int
    outgoing_transactions: int
    transactions_by_source: Dict[str, int]
    transactions_by_user: Dict[str, int]
    daily_summary: List[Dict[str, Any]]

# ----------------------------
# Utility Schemas
# ----------------------------
class PaginationParams(BaseModel):
    """Schema for pagination parameters"""
    skip: int = Field(0, ge=0, alias="offset", description="Number of records to skip")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of records to return")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[Literal["asc", "desc"]] = Field("desc", description="Sort order")

class SearchQuery(BaseModel):
    """Schema for search queries"""
    q: Optional[str] = Field(None, description="Search query")
    category: Optional[str] = None
    in_stock_only: bool = False
    active_only: bool = True

class SuccessResponse(BaseModel):
    """Standard success response schema"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    """Standard error response schema"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class BulkOperationResult(BaseModel):
    """Schema for bulk operation results"""
    total: int
    successful: int
    failed: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)

# ----------------------------
# Export/Import Schemas
# ----------------------------
class ProductImport(BaseModel):
    """Schema for product imports"""
    sku: str
    name: str
    category: Optional[str] = None
    quantity: Optional[int] = 0
    price: Optional[Decimal] = None
    cost: Optional[Decimal] = None
    
    supplier: Optional[str] = None
    location: Optional[str] = None
    tags: Optional[List[str]] = None

class ImportResult(BaseModel):
    """Schema for import results"""
    total_records: int
    imported: int
    skipped: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)