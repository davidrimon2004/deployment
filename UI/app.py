import os
import streamlit as st
import pandas as pd
import requests
from requests.exceptions import RequestException

# Read backend URL from environment (set this in Streamlit Cloud)
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.title("Retail Sales Predictor (Store-based Models)")

mode = st.radio("Choose Mode", ["Single Prediction", "Batch Prediction"])

if mode == "Single Prediction":
    store = st.selectbox("Store ID", ["CA_1","CA_2","CA_3","CA_4","TX_1","TX_2","TX_3","WI_1","WI_2","WI_3"])
    item = st.text_input("Item ID")
    price = st.number_input("Sell Price", value=0.0)
    wday = st.number_input("Weekday (1-7)", min_value=1, max_value=7, value=1)
    is_event = st.selectbox("Event Today?", [0,1])
    event_cnt = st.number_input("Event Count", min_value=0, value=0)
    lag1 = st.number_input("lag_1", value=0.0)
    lag7 = st.number_input("lag_7", value=0.0)
    
    # Additional features for the model
    wm_yr_wk = st.number_input("Week of Year", value=1)
    snap = st.selectbox("SNAP Day?", [0, 1])
    year = st.number_input("Year", value=2025)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    # Parse item_id into category/subcategory/number (simple heuristic)
    def parse_item_id(item_id: str):
        """Parse item_id in the format CATEGORY_subcat_number, e.g. FOODS_1_1.

        Returns (item_category, item_subcategory, item_number) as integers.
        - item_category: deterministic numeric mapping of the category string (simple hash)
        - item_subcategory: integer parsed from the second token if numeric, else hashed
        - item_number: integer parsed from the third token (or 0)
        """
        if not item_id or not isinstance(item_id, str):
            return 0, 0, 0

        parts = item_id.split("_")
        # category token (e.g., 'FOODS')
        cat_token = parts[0] if len(parts) >= 1 else "0"
        sub_token = parts[1] if len(parts) >= 2 else "0"
        num_token = parts[2] if len(parts) >= 3 else "0"

        # convert numeric-looking tokens to ints when possible
        try:
            item_number = int(num_token)
        except Exception:
            item_number = 0

        try:
            item_subcategory = int(sub_token)
        except Exception:
            # fallback: hash the string to an int
            item_subcategory = sum(ord(c) for c in str(sub_token)) % 1000

        # map category string to a stable numeric id
        def category_to_int(s: str) -> int:
            return sum(ord(c) for c in str(s)) % 1000

        item_category = category_to_int(cat_token)
        return item_category, item_subcategory, item_number

    item_category, item_subcategory, item_number = parse_item_id(item)

    # Computed features (do not let user input these)
    # price_flag: 1 if price < 0 else 0 (adjust threshold if needed)
    price_flag = 1 if price < 0 else 0
    # is_weekend: 1 if wday is Saturday(1) or Sunday(2)
    is_weekend = 1 if wday in (1, 2) else 0
    snap_weekend = int(snap) * int(is_weekend)
    # wday_x_snap: interaction between weekday and snap indicator
    wday_x_snap = int(wday) * int(snap)
    # event impact (user-provided) and use `is_event` directly
    event_impact = st.number_input("Event Impact", value=0.0)

    if st.button("Predict"):
        payload = {
            "store_id": store,
            "item_id": item,
            "sell_price": price,
            "wday": wday,
            "is_event_day": is_event,
            "event_count": event_cnt,
            "lag_1": lag1,
            "lag_7": lag7
        }
        payload = {
            "store_id": store,
            "wm_yr_wk": wm_yr_wk,
            "wday": wday,
            "snap": snap,
            "year": year,
            "month": month,
            "day": day,
            "lag_1": lag1,
            "item_category": item_category,
            "item_subcategory": item_subcategory,
            "item_number": item_number,
            "sell_price": price,
            "price_flag": price_flag,
            "is_weekend": is_weekend,
            "snap_weekend": snap_weekend,
            "wday_x_snap": wday_x_snap,
            "lag_7": lag7,
            "is_event": is_event,
            "event_count": event_cnt,
            "event_impact": event_impact
        }
        try:
            r = requests.post(API_URL + "/predict/single", json=payload, timeout=10)
            r.raise_for_status()
            st.success(f"Prediction: {r.json().get('prediction')}")
        except RequestException as e:
            st.error(f"Could not reach backend at {API_URL}: {e}")


else:
    file = st.file_uploader("Upload CSV", type="csv")

    if file is not None:
        if st.button("Predict Batch"):
            files = {"file": file}
            r = requests.post(API_URL + "/predict/batch", files=files)
            st.dataframe(pd.DataFrame(r.json()["predictions"]))
