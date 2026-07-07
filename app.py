# ==========================================
# Import Libraries
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Load Saved Files
# ==========================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#00E5FF;
    text-align:center;
}

h2,h3{
    color:white;
}

div[data-testid="stMetric"]{
    background:#F8FAFC;
    padding:20px;
    border-radius:15px;
    border:2px solid #00BCD4;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}
div[data-testid="stMetric"] label{
    color:#0F172A !important;
    font-size:18px !important;
    font-weight:bold;
}

div[data-testid="stMetric"] div{
    color:#0F172A !important;
}            

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    background:#00C853;
    color:white;
    font-size:20px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#009624;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar
# ==========================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1040/1040230.png",
    width=120
)

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Project Information

This application predicts
House Price using a Machine Learning Model.

### Model

✅ Random Forest Regressor

### Developer

**Sumit Ghodke**

AI Engineer | Data Scientist
"""
)

# ==========================================
# Header
# ==========================================

st.title("🏠 House Price Prediction System")

st.markdown(
"""
### Predict the estimated price of a residential property using Machine Learning.

Fill all property details below and click **Predict Price**.
"""
)

st.markdown("---")
# ==========================================
# User Input Section
# ==========================================

st.header("📝 Property Details")

col1, col2 = st.columns(2)

# -------------------------------
# Column 1
# -------------------------------

with col1:

    posted_by = st.selectbox(
        "Posted By",
        ["Owner", "Dealer", "Builder"]
    )

    under_construction = st.selectbox(
        "Under Construction",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    rera = st.selectbox(
        "RERA Approved",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    bhk_no = st.number_input(
        "Number of BHK",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    bhk_or_rk = st.selectbox(
        "Property Type",
        ["BHK", "RK"]
    )

    square_ft = st.number_input(
        "Square Feet",
        min_value=100.0,
        max_value=300000000.0,
        value=1200.0,
        step=50.0
    )

# -------------------------------
# Column 2
# -------------------------------

with col2:

    ready_to_move = st.selectbox(
        "Ready To Move",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    resale = st.selectbox(
        "Resale Property",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    longitude = st.number_input(
        "Longitude",
        value=77.59,
        format="%.6f"
    )

    latitude = st.number_input(
        "Latitude",
        value=12.97,
        format="%.6f"
    )

    city = st.text_input(
        "City",
        placeholder="Example : Bangalore"
    )

# --------------------------------
# Predict Button
# --------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button("🔍 Predict House Price")
# ==========================================
# Prediction Preprocessing
# ==========================================

if predict_button:

    # -----------------------------
    # Create Input DataFrame
    # -----------------------------

    input_df = pd.DataFrame({

        "POSTED_BY": [posted_by],
        "UNDER_CONSTRUCTION": [under_construction],
        "RERA": [rera],
        "BHK_NO": [bhk_no],
        "BHK_OR_RK": [bhk_or_rk],
        "READY_TO_MOVE": [ready_to_move],
        "RESALE": [resale],
        "LONGITUDE": [longitude],
        "LATITUDE": [latitude],
        "CITY": [city.title()],
        "SQUARE_FT_LOG": [np.log1p(square_ft)]

    })

    # ---------------------------------
    # Handle Rare / Unknown Cities
    # ---------------------------------

    common_cities = [

        "Bangalore","Lalitpur","Mumbai","Pune","Noida",
        "Kolkata","Maharashtra","Chennai","Ghaziabad",
        "Jaipur","Chandigarh","Faridabad","Mohali",
        "Vadodara","Gurgaon","Surat","Nagpur",
        "Lucknow","Indore","Bhubaneswar"

    ]

    if input_df.loc[0, "CITY"] not in common_cities:
        input_df.loc[0, "CITY"] = "Other"

    # ---------------------------------
    # One-Hot Encoding
    # ---------------------------------

    input_df = pd.get_dummies(input_df)

    # ---------------------------------
    # Match Training Columns
    # ---------------------------------

    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # ---------------------------------
    # Scale Numeric Columns
    # ---------------------------------

    numeric_features = [

        "UNDER_CONSTRUCTION",
        "RERA",
        "BHK_NO",
        "READY_TO_MOVE",
        "RESALE",
        "LONGITUDE",
        "LATITUDE",
        "SQUARE_FT_LOG"

    ]

    input_df[numeric_features] = scaler.transform(
        input_df[numeric_features]
    )

    # ---------------------------------
    # Ready for Prediction
    # ---------------------------------

    prediction_log = model.predict(input_df)[0]

    prediction_price = np.expm1(prediction_log)
    # ==========================================
# Prediction Result
# ==========================================

    st.markdown("---")

    st.success("Prediction Completed Successfully!")

    st.markdown("## 🏡 Estimated House Price")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="💰 Predicted Price",
            value=f"₹ {prediction_price:,.2f} Lakhs"
        )

    with col2:
        st.metric(
            label="🏠 Property Type",
            value=bhk_or_rk
        )

    with col3:
        st.metric(
            label="📐 Square Feet",
            value=f"{square_ft:,.0f}"
        )

    st.markdown("---")

    st.subheader("📋 Property Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Posted By",
            "City",
            "BHK",
            "Property Type",
            "Square Feet",
            "Under Construction",
            "Ready To Move",
            "Resale",
            "RERA Approved",
            "Longitude",
            "Latitude"

        ],

        "Value":[

            posted_by,
            city,
            bhk_no,
            bhk_or_rk,
            square_ft,
            "Yes" if under_construction else "No",
            "Yes" if ready_to_move else "No",
            "Yes" if resale else "No",
            "Yes" if rera else "No",
            longitude,
            latitude

        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.info(
        f"""
### 📊 Prediction Result

Based on the property details entered,

### ✅ Estimated House Price

# ₹ {prediction_price:,.2f} Lakhs

This prediction is generated using the trained **Random Forest Regression Model**.
"""
    )
# ==========================================
# Download Prediction Report
# ==========================================

    report = pd.DataFrame({

        "Feature":[

            "Posted By",
            "City",
            "BHK",
            "Property Type",
            "Square Feet",
            "Under Construction",
            "Ready To Move",
            "Resale",
            "RERA Approved",
            "Longitude",
            "Latitude",
            "Predicted Price (Lakhs)"

        ],

        "Value":[

            posted_by,
            city,
            bhk_no,
            bhk_or_rk,
            square_ft,
            under_construction,
            ready_to_move,
            resale,
            rera,
            longitude,
            latitude,
            round(prediction_price,2)

        ]

    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Prediction Report",

        data=csv,

        file_name="House_Price_Prediction.csv",

        mime="text/csv"

    )

# ==========================================
# Model Information
# ==========================================

st.markdown("---")

with st.expander("ℹ️ Model Information", expanded=False):

    st.write("""

### Machine Learning Model

- Model : Random Forest Regressor
- Problem Type : Regression
- Target Variable : House Price (Lakhs)
- Evaluation Metric : R² Score

### Performance

- Best Test R² : **0.9356**
- Cross Validation Mean R² : **0.8490**
- Cross Validation Std : **0.0065**

### Features Used

- Posted By
- Under Construction
- RERA
- BHK Number
- BHK / RK
- Square Feet
- Ready To Move
- Resale
- Longitude
- Latitude
- City

""")

# ==========================================
# Tips
# ==========================================

st.markdown("---")

st.subheader("💡 Tips")

st.info("""

✔ Enter realistic property details.

✔ Use correct longitude and latitude values.

✔ Enter the actual city name.

✔ The prediction is an estimated market price.

✔ The model performs best on properties similar to the training data.

""")

# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.markdown("""

<div style="text-align:center">

<h3>🏠 House Price Prediction using Machine Learning</h3>

<p>

Developed using

<b>Python • Pandas • NumPy • Scikit-Learn • Streamlit</b>

</p>

<p>

Machine Learning Model:
<b>Random Forest Regressor</b>

</p>

<p>

Developed by <b>Sumit Ghodke</b>

</p>

</div>

""", unsafe_allow_html=True)

import os
import urllib.request
import joblib

MODEL_URL = "https://huggingface.co/Sumitghodke74kg/house-price-randomforest-model/resolve/main/random_forest_model.pkl"

MODEL_PATH = "random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = joblib.load(MODEL_PATH)