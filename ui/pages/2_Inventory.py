import streamlit as st
from api.products import list_products
from api.transactions import create_transaction

st.header("📦 Inventory Adjustment")
st.caption("Increment or decrement product stock")

# Load products
try:
    products = list_products()
except Exception:
    st.error("Failed to load products")
    st.stop()

if not products:
    st.warning("No products available")
    st.stop()

# Product selection
product_map = {f"{p['sku']} — {p['name']}": p for p in products}
selection = st.selectbox("Select Product", list(product_map.keys()))
product = product_map[selection]

st.info(f"Current Quantity: **{product['quantity']}**")

# Adjustment form
with st.form("inventory_adjustment_form"):
    action = st.radio("Action", ["INCREMENT", "DECREMENT"], horizontal=True)
    quantity = st.number_input("Quantity", min_value=1, step=1)
    note = st.text_input("Note (optional)")
    submitted = st.form_submit_button("Apply Change")

# Submit logic
if submitted:
    delta = quantity if action == "INCREMENT" else -quantity

    payload = {
        "product_sku": product["sku"],
        "transaction_type": action,
        "quantity": delta,
        "note": note,
        "source": "UI"
    }

    try:
        result = create_transaction(payload)
        st.success("Inventory updated successfully")
        st.json(result)
    except Exception as e:
        st.error("Failed to update inventory")
