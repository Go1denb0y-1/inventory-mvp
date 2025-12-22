import streamlit as st
import pandas as pd
from datetime import date

from api.transactions import list_transactions

st.set_page_config(page_title="Transaction History", layout="wide")

st.title("Transaction History")

# -----------------------------
# Filters
# -----------------------------

with st.sidebar:
    st.header("Filters")

    sku_filter = st.text_input("Product SKU")
    tx_type = st.selectbox(
        "Transaction Type",
        options=["All", "IN", "OUT"]
    )

    start_date = st.date_input("Start Date", value=None)
    end_date = st.date_input("End Date", value=None)

    apply_filters = st.button("Apply Filters")

# -----------------------------
# Fetch data
# -----------------------------

params = {}

if apply_filters:
    if sku_filter:
        params["product_sku"] = sku_filter.upper()

    if tx_type != "All":
        params["transaction_type"] = tx_type

    if start_date:
        params["start_date"] = start_date.isoformat()

    if end_date:
        params["end_date"] = end_date.isoformat()

transactions = list_transactions(params if params else None)

# -----------------------------
# Display
# -----------------------------

if not transactions:
    st.info("No transactions found.")
    st.stop()

df = pd.DataFrame(transactions)

# Ensure consistent column order
columns = [
    "id",
    "product_sku",
    "transaction_type",
    "quantity",
    "source",
    "note",
    "timestamp",
]

df = df[[c for c in columns if c in df.columns]]

st.dataframe(df, width="stretch")

# -----------------------------
# Export
# -----------------------------

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Export CSV",
    data=csv,
    file_name="transaction_history.csv",
    mime="text/csv"
)
