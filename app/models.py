from sqlalchemy import Column, Integer, String, Numeric, DateTime, ARRAY, CheckConstraint, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    category = Column(String(100))
    quantity = Column(Integer, nullable=False, default=0)
    rfid_tag = Column(String(100), unique=True)
    price = Column(Numeric(10,2))
    tags = Column(ARRAY(String))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    product_sku = Column(String(50), ForeignKey("products.sku"))
    transaction_type = Column(String(3), CheckConstraint("transaction_type IN ('IN','OUT')"))
    quantity = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    source = Column(String(50), default='MANUAL')
    note = Column(String)

class InventoryHistory(Base):
    __tablename__ = "inventory_history"

    id = Column(Integer, primary_key=True, index=True)
    product_sku = Column(String, ForeignKey("products.sku"))
    change_type = Column(String)  # e.g. "RFID_IN", "RFID_OUT", "TRANSACTION_IN", etc.
    delta = Column(Integer)
    action = Column(String, nullable=False)
    quantity_before = Column(Integer)
    quantity_after = Column(Integer)
    source = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product")