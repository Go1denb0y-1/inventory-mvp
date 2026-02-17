from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, ARRAY, 
    CheckConstraint, ForeignKey, Float, Index, Boolean, Enum
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from app.database import Base
import enum


# ----------------------------
# Enums for better data integrity
# ----------------------------
class TransactionType(str, enum.Enum):
    IN = "IN"
    OUT = "OUT"

class ChangeType(str, enum.Enum):
    RFID_IN = "RFID_IN"
    RFID_OUT = "RFID_OUT"
    MANUAL_IN = "MANUAL_IN"
    MANUAL_OUT = "MANUAL_OUT"
    ADJUSTMENT = "ADJUSTMENT"
    CORRECTION = "CORRECTION"
    TRANSFER_IN = "TRANSFER_IN"
    TRANSFER_OUT = "TRANSFER_OUT"

class SourceType(str, enum.Enum):
    MANUAL = "MANUAL"
    RFID = "RFID"
    API = "API"
    SYSTEM = "SYSTEM"
    IMPORT = "IMPORT"


# ----------------------------
# Product Model
# ----------------------------
class Product(Base):
    __tablename__ = "products"

    # Primary key (necessary for SQLAlchemy)
    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(50), unique=True, index=True, doc="Unique product SKU")
    
    name = Column(String(255), nullable=False, index=True, doc="Product name")
    category = Column(String(100), nullable=True, index=True, doc="Product category")
    quantity = Column(Integer, nullable=False, default=0, doc="Product quantity in stock")
    
    rfid_tag = Column(String(100), nullable=True, unique=True, index=True, doc="RFID tag if available")
    price = Column(Numeric(10, 2), nullable=True, doc="Selling price")
    cost = Column(Numeric(10, 2), nullable=True, doc="Cost price")
    
    tags = Column(ARRAY(String), default=[], doc="List of product tags")
    location = Column(String(100), nullable=True, doc="Storage location")
    supplier = Column(String(100), nullable=True, doc="Product supplier")
    
    is_active = Column(Boolean, nullable=False, default=True, index=True, doc="Product active status")
    
    # Extra fields from ProductBase
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), doc="Last update timestamp")
    source_system = Column(String(50), default="Inventory_system", nullable=False, doc="Origin system of the product")

    # Relationships
    transactions = relationship(
        "Transaction",
        back_populates="product",
        cascade="all, delete-orphan",
        lazy="dynamic",
        foreign_keys="Transaction.product_sku"
    )
    
    history_entries = relationship(
        "InventoryHistory",
        back_populates="product",
        cascade="all, delete-orphan",
        lazy="dynamic",
        foreign_keys="InventoryHistory.product_sku"
    )

    # Table-level constraints
    __table_args__ = (
        CheckConstraint('quantity >= 0', name='non_negative_quantity'),
        CheckConstraint('price >= 0 OR price IS NULL', name='non_negative_price'),
        CheckConstraint('cost >= 0 OR cost IS NULL', name='non_negative_cost'),
        Index('idx_product_sku_active', 'sku', 'is_active'),
        Index('idx_product_rfid_active', 'rfid_tag', 'is_active'),
        Index('idx_product_category_quantity', 'category', 'quantity'),
    )

    # Computed properties
    @property
    def total_value(self):
        return float(self.price or 0) * self.quantity

    @property
    def profit_margin(self):
        if self.cost and self.price and self.cost > 0:
            return float((self.price - self.cost) / self.cost) * 100
        return None

    def __repr__(self):
        return f"<Product(sku='{self.sku}', name='{self.name}', quantity={self.quantity})>"

# ----------------------------
# Transaction Model
# ----------------------------
class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Use Integer foreign key for better performance (optional)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), index=True)
    product_sku = Column(String(50), ForeignKey("products.sku", ondelete="CASCADE"), index=True)
    
    transaction_type = Column(
        Enum(TransactionType),
        nullable=False,
        index=True,
        doc="Type of transaction: IN or OUT"
    )
    
    quantity = Column(
        Integer, 
        nullable=False,
        doc="Quantity moved in this transaction"
    )
    
    # Store both to avoid joins for common queries
    unit_price = Column(Numeric(10, 2), doc="Price per unit at time of transaction")
    total_value = Column(Numeric(12, 2), doc="Total value of transaction")
    
    timestamp = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    source = Column(
        Enum(SourceType),
        default=SourceType.MANUAL,
        nullable=False,
        index=True,
        doc="Source of the transaction"
    )
    
    reference = Column(String(100), doc="External reference number")
    note = Column(String(500), doc="Additional notes")
    
    user_id = Column(String(100), doc="User who performed the transaction")
    device_id = Column(String(100), doc="Device used for the transaction")
    
    # Relationships
    product = relationship(
        "Product",
        back_populates="transactions",
        foreign_keys=[product_sku]  # Specify foreign key to avoid ambiguity
    )
    
    # Table-level constraints
    __table_args__ = (
        CheckConstraint('quantity > 0', name='positive_quantity'),
        CheckConstraint('unit_price >= 0 OR unit_price IS NULL', name='non_negative_unit_price'),
        CheckConstraint('total_value >= 0 OR total_value IS NULL', name='non_negative_total_value'),
        Index('idx_transaction_date_type', 'timestamp', 'transaction_type'),
        Index('idx_transaction_product_date', 'product_id', 'timestamp'),
        Index('idx_transaction_source_date', 'source', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Transaction(product={self.product_sku}, type={self.transaction_type}, qty={self.quantity})>"


# ----------------------------
# Inventory History Model
# ----------------------------
class InventoryHistory(Base):
    __tablename__ = "inventory_history"
    
    id = Column(Integer, primary_key=True)
    
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), index=True)
    product_sku = Column(String(50), ForeignKey("products.sku", ondelete="CASCADE"), index=True)
    
    change_type = Column(
        Enum(ChangeType),
        nullable=False,
        index=True,
        doc="Type of inventory change"
    )
    
    # Don't store delta - compute it if needed
    quantity_before = Column(Integer, nullable=False, doc="Quantity before change")
    quantity_after = Column(Integer, nullable=False, doc="Quantity after change")
    
    # Reference to transaction if applicable
    transaction_id = Column(
        Integer, 
        ForeignKey("transactions.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    source = Column(
        Enum(SourceType),
        nullable=True,
        index=True,
        doc="Source of the change"
    )
    
    reason = Column(String(500), doc="Reason for the change")
    
    performed_by = Column(String(100), doc="Who performed the change")
    
    timestamp = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    # Relationships
    product = relationship(
        "Product",
        back_populates="history_entries",
        foreign_keys=[product_sku]
    )
    
    transaction = relationship(
        "Transaction",
        backref=backref("history_entry", uselist=False),  # One-to-one if needed
        foreign_keys=[transaction_id]
    )
    
    # Table-level constraints
    __table_args__ = (
        CheckConstraint('quantity_before >= 0', name='non_negative_qty_before'),
        CheckConstraint('quantity_after >= 0', name='non_negative_qty_after'),
        Index('idx_history_product_date', 'product_id', 'timestamp'),
        Index('idx_history_change_type_date', 'change_type', 'timestamp'),
        Index('idx_history_product_change', 'product_id', 'change_type', 'timestamp'),
    )
    
    # Computed property
    @property
    def delta(self):
        """Compute the change in quantity"""
        return self.quantity_after - self.quantity_before
    
    def __repr__(self):
        return f"<InventoryHistory(product={self.product_sku}, change={self.change_type}, delta={self.delta})>"


# ----------------------------
# Optional: RFID Tag Model
# ----------------------------
class RFIDTag(Base):
    __tablename__ = "rfid_tags"
    
    id = Column(Integer, primary_key=True, index=True)
    tag_uid = Column(String(100), unique=True, nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"))
    product_sku = Column(String(50), ForeignKey("products.sku", ondelete="CASCADE"))
    
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_seen = Column(DateTime(timezone=True), doc="Last time tag was scanned")
    first_seen = Column(DateTime(timezone=True), server_default=func.now())
    
    location = Column(String(100), doc="Last known location")
    device_id = Column(String(100), doc="Device that last scanned the tag")
    
    notes = Column(String(500))
    
    # Relationship
    product = relationship(
        "Product", 
        backref="rfid_tags",  
        foreign_keys=[product_sku]
        )
    
    __table_args__ = (
        Index('idx_rfid_active', 'tag_uid', 'is_active'),
    )
    
    def __repr__(self):
        return f"<RFIDTag(uid={self.tag_uid}, product={self.product_sku})>"