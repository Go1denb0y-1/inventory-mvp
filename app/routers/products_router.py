from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product
from app.schemas import ProductCreate, ProductOut

router = APIRouter(
    prefix="/products",
    tags=["Products"]
)

@router.post("/", response_model=ProductOut)
def create_product(payload: ProductCreate, db: Session = Depends(get_db)):
    sku = payload.sku.upper()

    existing = db.query(Product).filter(Product.sku == sku).first()
    if existing:
        raise HTTPException(status_code=400, detail="Product already exists")

    product = Product(
        **payload.dict(exclude_unset=True),
        sku=sku
    )

    db.add(product)
    db.commit()
    db.refresh(product)
    return product

@router.get("/", response_model=list[ProductOut])
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).all()

@router.get("/{sku}", response_model=ProductOut)
def get_product(sku: str, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.sku == sku.upper()).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
