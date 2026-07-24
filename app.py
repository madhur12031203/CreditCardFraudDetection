import streamlit as st
import pandas as pd
import joblib
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

h1 {
    color: #2563eb;
    text-align: center;
    font-weight: 700;
}

h2,h3 {
    color:#1e293b;
}

[data-testid="stMetric"]{
    background:#ffffff;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    border-left:6px solid #2563eb;
}

.stButton>button{
    width:100%;
    background:#2563eb;
    color:white;
    border-radius:10px;
    height:3.2em;
    font-size:17px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#1d4ed8;
    color:white;
}

[data-testid="stSidebar"]{
    background:#0f172a;
}

[data-testid="stSidebar"] *{
    color:white;
}

div[data-testid="stDataFrame"]{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


# Load model
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.markdown("""
<div style="background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 5px 20px rgba(0,0,0,.08);
text-align:center;
margin-bottom:25px;">

<h1 style="color:#2563EB;margin-bottom:10px;">
🛡️ Credit Card Fraud Detection
</h1>

<p style="font-size:20px;color:#475569;margin-bottom:10px;">
Machine Learning Powered Fraud Detection System
</p>

<p style="color:#64748B;">
Predict fraudulent transactions using Logistic Regression,
SMOTE and Streamlit.
</p>

</div>
""", unsafe_allow_html=True)
st.sidebar.title("🧭 Navigation")

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Welcome 👋

Choose a prediction mode:

• 💳 Single Transaction

• 📂 Batch Prediction
"""
)

st.sidebar.markdown("---")

# Sidebar
mode = st.sidebar.radio(
    "Choose Prediction Mode",
    ["Single Transaction", "Batch Prediction"]
)

# -------------------------
# SINGLE TRANSACTION
# -------------------------
if mode == "Single Transaction":

    st.header("💳 Single Transaction Prediction")


    # Basic Inputs
    time = st.number_input("Time", value=0.0)
    amount = st.number_input("Amount", value=0.0)

    # Advanced Features
    with st.expander("⚙ Advanced Features (V1 - V28)"):

        features = []

        for i in range(1, 29):
            value = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f",
                key=f"v{i}"
            )
            features.append(value)

    if st.button("🔍 Predict"):

        data = [[time] + features + [amount]]

        df_input = pd.DataFrame(
            data,
            columns=[
                "Time",
                *[f"V{i}" for i in range(1,29)],
                "Amount"
            ]
        )

        X_scaled = scaler.transform(df_input)

        prediction = model.predict(X_scaled)[0]

        probability = model.predict_proba(X_scaled)[0][1]

        # Display Result
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        if prediction == 1:
            col1.error("🚨 Fraud Detected")
        else:
            col1.success("✅ Genuine Transaction")

        col2.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
            )

        st.progress(float(probability))


# -------------------------
# BATCH PREDICTION
# -------------------------
elif mode == "Batch Prediction":

    st.write(
        "Upload a CSV file containing credit card transactions."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("📋 Uploaded Dataset")
        st.dataframe(df.head())

        if st.button("🔍 Predict Fraud"):

            if "Class" in df.columns:
                X = df.drop("Class", axis=1)
            else:
                X = df.copy()

            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:,1]

            df["Prediction"] = predictions

            df["Prediction"] = df["Prediction"].map({
                0:"✅ Genuine",
                1:"🚨 Fraud"
            })

            df["Fraud Probability"] = (
                probabilities*100
            ).round(2).astype(str)+"%"

            st.success("Prediction Completed!")
            st.balloons()

            frauds = (predictions==1).sum()
            genuine = (predictions==0).sum()

            col1,col2,col3 = st.columns(3)

            col1.metric("Transactions",len(df))
            col2.metric("Frauds",frauds)
            col3.metric("Genuine",genuine)

            st.subheader("📊 Prediction Results")

            st.dataframe(
                df[
                    [
                        "Time",
                        "Amount",
                        "Prediction",
                        "Fraud Probability"
                    ]
                ]
            )

            chart_data = pd.DataFrame({
                "Category": ["Genuine", "Fraud"],
                "Count": [genuine, frauds]
                    })

            st.subheader("📈 Prediction Summary")
            st.bar_chart(chart_data.set_index("Category"))
            csv = df.to_csv(index=False).encode("utf-8")
            if st.checkbox("Show Only Fraud Transactions"):
                 st.dataframe(
                     df[df["Prediction"] == "🚨 Fraud"]
                 )

            st.download_button(
                label="📥 Download Results",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
                )

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:gray">

### 🤖 Model Information

**Algorithm:** Logistic Regression

**Preprocessing:** StandardScaler + SMOTE

**Framework:** Streamlit

© 2026 | Developed by <b>Madhur</b>

</div>
""", unsafe_allow_html=True)