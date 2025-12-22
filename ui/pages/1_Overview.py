import streamlit as st
import pandas as pd

from api.products import list_products
from api.transactions import list_transactions  # you will create this

st.header("📊 Overview Dashboard")
st.caption("High-level snapshot of inventory health")

try:
    products = list_products()
except Exception as e:
    st.error("Failed to load products")
    st.stop()

try:
    transactions = list_transactions()
except Exception:
    transactions = []

df_products = pd.DataFrame(products)

total_products = len(df_products)

total_stock_value = 0
if not df_products.empty and "price" in df_products.columns:
    total_stock_value = (df_products["quantity"] * df_products["price"].fillna(0)).sum()

low_stock_df = df_products[df_products["quantity"] <= 5]

col1, col2, col3 = st.columns(3)

col1.metric("Total Products", total_products)
col2.metric("Total Stock Value", f"${total_stock_value:,.2f}")
col3.metric("Low Stock Items", len(low_stock_df))

st.subheader("⚠️ Low Stock Alerts")

if low_stock_df.empty:
    st.success("No low-stock items")
else:
    st.dataframe(
        low_stock_df[["sku", "name", "quantity"]],
        use_container_width=True
    )

st.subheader("🕒 Recent Transactions")

if transactions:
    df_tx = pd.DataFrame(transactions).sort_values("id", ascending=False).head(10)
    st.dataframe(df_tx, use_container_width=True)
else:
    st.info("No transactions recorded yet")
