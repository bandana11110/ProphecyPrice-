import streamlit as st
import numpy as np
from joblib import load

# Configuration

st.set_page_config(
    page_title="ProphecyPrice🏡",
    page_icon="🏠",
    layout="wide"
)

# Model

model = load("Dragon.joblib")

# Dashboard

st.title(" ProphecyPrice🏡")
st.markdown("""
### Predict Property Values with Machine Learning

Built using Scikit-Learn, Data Analysis, Feature Engineering,
and Machine Learning Pipelines.

Get instant property value predictions based on
location, environmental, and economic factors.
""")

st.divider()

st.info("""

This application predicts housing prices using 13 housing-market features.

The model was trained using Machine Learning techniques and deployed using Streamlit.
""")

with st.expander("📚 Feature Guide"):
    st.write("CRIM → Crime rate by town")
    st.write("ZN → Residential land proportion")
    st.write("INDUS → Non-retail business area proportion")
    st.write("CHAS → Charles River proximity (0 or 1)")
    st.write("NOX → Nitric oxide concentration")
    st.write("RM → Average number of rooms")
    st.write("AGE → Proportion of older houses")
    st.write("DIS → Distance to employment centers")
    st.write("RAD → Accessibility to highways")
    st.write("TAX → Property tax rate")
    st.write("PTRATIO → Pupil-teacher ratio")
    st.write("B → Demographic feature")
    st.write("LSTAT → Lower-status population percentage")

st.divider()

m1, m2, m3 = st.columns(3)

m1.metric("Features", "13")
m2.metric("Model", "Random Forest")
m3.metric("RMSE", "2.94")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Location")

    crim = st.number_input("CRIM (Crime Rate)", value=0.08)
    zn = st.number_input("ZN (Residential Land)", value=0.0)
    indus = st.number_input("INDUS", value=0.0)
    chas = st.selectbox("CHAS", [0, 1])

with col2:
    st.subheader("🌍 Environment")

    nox = st.number_input("NOX", value=0.5)
    rm = st.number_input("RM (Rooms)", value=6.0)
    age = st.number_input("AGE", value=50.0)
    dis = st.number_input("DIS", value=4.0)

with col3:
    st.subheader("📊 Economics")

    rad = st.number_input("RAD", value=1)
    tax = st.number_input("TAX", value=300)
    ptratio = st.number_input("PTRATIO", value=15.0)
    b = st.number_input("B", value=390.0)
    lstat = st.number_input("LSTAT", value=10.0)

st.divider()

# Prediction

if st.button("🔮 Predict Property Value", use_container_width=True):

    features = np.array([[
        crim,
        zn,
        indus,
        chas,
        nox,
        rm,
        age,
        dis,
        rad,
        tax,
        ptratio,
        b,
        lstat
    ]])

    prediction = model.predict(features)

    st.metric(
        "🏡 Estimated Property Value",
        f"${prediction[0]:.2f}K"
    )

    if prediction[0] >= 35:
        st.success("🌟 Luxury Property")
    elif prediction[0] >= 20:
        st.info("🏠 Mid-Range Property")
    else:
        st.warning("💰 Affordable Property")

    st.balloons()

# Sidebar

with st.sidebar:

    st.title("ProphecyPrice🏡")

    st.write("""
    AI-Powered Boston Housing Price Prediction
    """)

    st.divider()

    st.subheader("🚀 Project")

    st.write("""

    Boston Housing Price Prediction
    using Scikit-Learn and Streamlit.
    """)

    st.divider()

    st.subheader("🛠️ Tech Stack")

    st.write("""
    • Python
    • NumPy
    • Pandas
    • Scikit-Learn
    • Streamlit
    • Joblib
    """)

    st.divider()

    st.subheader("📊 Model Details")

    st.write("""
    Features: 13

    Target: MEDV

    Algorithm: Random Forest

    RMSE: 2.94
    """)

    st.divider()

    st.sidebar.markdown("""
    ##👨‍💻 Developer

    Bandana Tripathy

    First Machine Learning Project 🚀
    
    built with Python ,scikit-learn and streamlit.
    """)