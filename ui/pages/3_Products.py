import streamlit as st
import pandas as pd

from api.products import (
    list_products,
    create_product,
    update_product
)

st.header("🛒 Product Management")
st.caption("Create, edit, and manage inventory products")

# ---------------- Load Products ----------------
try:
    products = list_products()
except Exception:
    st.error("Failed to load products")
    st.stop()

# ---------------- Create Product ----------------
st.subheader("➕ Add New Product")

with st.form("create_product_form"):
    sku = st.text_input("SKU (optional)")
    name = st.text_input("Product Name")
    category = st.text_input("Category")
    quantity = st.number_input("Initial Quantity", min_value=0, step=1)
    price = st.number_input("Price", min_value=0.0, step=0.01)
    rfid_tag = st.text_input("RFID Tag (optional)")
    submitted = st.form_submit_button("Create Product")

if submitted:
    if not name:
        st.warning("Product name is required")
    else:
        payload = {
            "sku": sku or None,
            "name": name,
            "category": category,
            "quantity": quantity,
            "price": price,
            "rfid_tag": rfid_tag or None
        }

        try:
            create_product(payload)
            st.success("Product created successfully")
            st.experimental_rerun()
        except Exception as e:
            st.error("Failed to create product")

# ---------------- Product Table ----------------
st.divider()
st.subheader("📋 Product List")

if not products:
    st.info("No products available")
    st.stop()

df = pd.DataFrame(products)
st.dataframe(df, use_container_width=True)

# ---------------- Edit Product ----------------
st.divider()
st.subheader("✏️ Edit Product")

product_map = {f"{p['sku']} — {p['name']}": p for p in products}
selection = st.selectbox("Select Product", list(product_map.keys()))
product = product_map[selection]

with st.form("edit_product_form"):
    name = st.text_input("Name", value=product["name"])
    category = st.text_input("Category", value=product.get("category") or "")
    price = st.number_input("Price", value=product.get("price") or 0.0, step=0.01)
    rfid_tag = st.text_input("RFID Tag", value=product.get("rfid_tag") or "")
    submitted = st.form_submit_button("Update Product")

if submitted:
    payload = {
        "name": name,
        "category": category,
        "price": price,
        "rfid_tag": rfid_tag or None
    }

    try:
        update_product(product["sku"], payload)
        st.success("Product updated")
        st.experimental_rerun()
    except Exception:
        st.error("Failed to update product")
