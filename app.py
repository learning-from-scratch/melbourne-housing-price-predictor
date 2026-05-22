app_code = '''
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load model
artifact = joblib.load("outputs/melbourne_housing_price_model.joblib")
model         = artifact["model"]
model_name    = artifact["model_name"]
feature_cols  = artifact["feature_columns"]

# Page config
st.set_page_config(page_title="Melbourne Housing Price Predictor")
st.title("Melbourne Housing Price Predictor")
st.markdown(
    f"Predict the sold price of a property in **Fitzroy**, **Carlton**, or **Mentone** "
    f"using the **{model_name}** model trained on realestate.com.au data."
)
st.divider()

# Sidebar inputs
st.sidebar.header("Property Details")

suburb = st.sidebar.selectbox(
    "Suburb", ["Fitzroy", "Carlton", "Mentone"]
)
property_type = st.sidebar.selectbox(
    "Property type", ["House", "Apartment", "Townhouse", "Studio", "Villa"]
)
sale_method = st.sidebar.selectbox(
    "Sale method", ["Auction", "Private treaty"]
)
street_type = st.sidebar.selectbox(
    "Street type",
    ["Street", "Road", "Avenue", "Lane", "Place", "Court", "Crescent", "Parade", "Drive", "Other"]
)
bedrooms   = st.sidebar.slider("Bedrooms",   0, 8, 2)
bathrooms  = st.sidebar.slider("Bathrooms",  0, 5, 1)
car_spaces = st.sidebar.slider("Car spaces", 0, 4, 1)
has_unit   = st.sidebar.checkbox("Has unit number (e.g. 3/10 Smith St)", value=False)
sold_month = st.sidebar.slider("Sale month", 1, 12, 5)
sold_year  = st.sidebar.selectbox("Sale year", [2024, 2025, 2026], index=1)

# Derived features
SUBURB_POSTCODES = {"Fitzroy": 3065, "Carlton": 3053, "Mentone": 3194}
postcode    = SUBURB_POSTCODES[suburb]
sold_quarter = (sold_month - 1) // 3 + 1

# Predict
input_df = pd.DataFrame([{
    "suburb":               suburb,
    "property_type_clean":  property_type,
    "sale_method":          sale_method,
    "street_type":          street_type,
    "bedrooms":             float(bedrooms),
    "bathrooms":            float(bathrooms),
    "car_spaces":           float(car_spaces),
    "sold_year":            float(sold_year),
    "sold_month":           float(sold_month),
    "sold_quarter":         float(sold_quarter),
    "postcode":             float(postcode),
    "has_unit_number":      int(has_unit),
}])

prediction = model.predict(input_df)[0]

# Display result
st.subheader("Estimated Sold Price")
st.metric(label=model_name, value=f"${prediction:,.0f}")
st.divider()

st.subheader("Your Input Summary")
summary = input_df.copy()
summary["Property has a unit/strata number in address (e.g. 3/10 Smith St)"] = "Yes" if has_unit else "No"
st.dataframe(summary.T.rename(columns={0: "Value"}), use_container_width=True)
'''

# Write app.py to disk
with open("app.py", "w") as f:
    f.write(app_code.strip())

print("app.py written successfully.")
print("To launch the app, run:  streamlit run app.py")