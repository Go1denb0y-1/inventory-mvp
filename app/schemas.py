from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class ProductBase(BaseModel):
    sku: Optional[str] = None
    name: str
    category: Optional[str] = None
    quantity: int = 0
    rfid_tag: Optional[str] = None
    price: Optional[float] = None
    tags: List[str] = Field(default_factory=list)

class ProductCreate(ProductBase):
    pass

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    tags: Optional[List[str]] = None
    rfid_tag: Optional[str] = None

class ProductOut(ProductBase):
    class Config:
        from_attributes = True

class RFIDScan(BaseModel):
    rfid_tag: str
    mode: Literal["IN", "OUT"]
    quantity: int = 1
    source: Optional[str] = None

class QuantityChange(BaseModel):
    delta: int
    note: Optional[str] = None

class TransactionCreate(BaseModel):
    product_sku: str
    transaction_type: Literal["IN", "OUT"]
    quantity: int
    note: Optional[str] = None
    source: Optional[str] = "MANUAL"

class TransactionOut(BaseModel):
    id: int
    product_sku: str
    transaction_type: str
    quantity: int
    note: Optional[str]
    source: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True

class InventoryHistoryOut(BaseModel):
    id: int
    product_sku: str
    change_type: str
    delta: int
    quantity_before: int
    quantity_after: int
    source: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True
