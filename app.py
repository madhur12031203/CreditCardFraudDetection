"""
app.py
------
Streamlit inference application for Credit Card Fraud Detection.
Redesigned with modern Glassmorphism Dark Theme, Plotly analytics,
and production-ready UI/UX architecture.
"""

import os
import sys
import io
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# PATH CONFIGURATION & CACHED ARTIFACT LOADING
# -----------------------------------------------------------------------------
MODEL_PATH = "fraud_detection_model.pkl"
SCALER_PATH = "scaler.pkl"


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Any, Any, List[str]]:
    """
    Loads saved model and scaler artifacts with Streamlit caching.
    Returns model, scaler, and a list of missing paths if any.
    """
    missing = [p for p in (MODEL_PATH, SCALER_PATH) if not os.path.exists(p)]
    if missing:
        return None, None, missing

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, []


# -----------------------------------------------------------------------------
# CUSTOM GLASSMORPHISM & UI STYLING
# -----------------------------------------------------------------------------
def inject_custom_css():
    """Injects bespoke dark-theme CSS with glassmorphic cards and polished components."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-main: #0b0f17;
        --bg-card: rgba(22, 31, 48, 0.65);
        --bg-card-hover: rgba(30, 42, 66, 0.85);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-bright: rgba(255, 255, 255, 0.18);
        --accent-blue: #2563eb;
        --accent-indigo: #6366f1;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.10) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(244, 63, 94, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(13, 19, 31, 0.88) !important;
        backdrop-filter: blur(16px);
        border-right: 1px solid var(--border-glass);
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.20) 0%, rgba(99, 102, 241, 0.15) 100%);
        border: 1px solid var(--border-glass-bright);
        border-radius: 24px;
        padding: 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        text-align: center;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Glass Cards */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    .glass-card:hover {
        border-color: var(--border-glass-bright);
        box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.4);
    }

    /* Metric Cards */
    .stat-card {
        background: rgba(22, 31, 48, 0.5);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f9fafb;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* Result Cards */
    .result-card-fraud {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.25) 0%, rgba(159, 18, 57, 0.35) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-card-genuine {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(4, 120, 87, 0.35) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Form & Input Overrides */
    div[data-baseweb="input"] > div {
        background-color: rgba(22, 31, 48, 0.8) !important;
        border-color: var(--border-glass) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #2563eb 0%, #4f46e5 100%) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.4) !important;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        color: var(--text-muted);
        font-size: 0.85rem;
        border-top: 1px solid var(--border-glass);
        margin-top: 3rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN APPLICATION LOGIC
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # Load machine learning artifacts
    model, scaler, missing = load_artifacts()

    # Handling Missing Artifact Error State
    if missing:
        st.error("⚠️ **Model Artifacts Not Found**")
        st.warning(
            f"Required model files missing: `{missing}`\n\n"
            "Please ensure `fraud_detection_model.pkl` and `scaler.pkl` exist in the root directory."
        )
        st.stop()

    # Session State Initialization for Recent Evaluations
    if "eval_history" not in st.session_state:
        st.session_state.eval_history = []

    # -------------------------------------------------------------------------
    # SIDEBAR: NAVIGATION & CONTROL
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("<h2 style='margin-bottom: 0;'>🛡️ Navigation</h2>", unsafe_allow_html=True)
        st.caption("AI-Powered Risk Assessment Engine")
        st.divider()

        mode = st.radio(
            "Select Mode:",
            ["💳 Single Transaction", "📂 Batch Prediction"],
            help="Choose between evaluating an individual transaction or uploading a dataset."
        )

        st.divider()
        st.markdown("### ⚙️ Risk Sensitivity")
        threshold = st.slider(
            "Fraud Decision Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Adjust probability cutoff for classifying a transaction as fraud."
        )

        st.divider()
        st.markdown("### ℹ️ Engine Specs")
        st.markdown("""
        - **Model:** Logistic Regression
        - **Sampling:** SMOTE Balance
        - **Scaler:** StandardScaler
        - **PCA Features:** V1 – V28
        """)

        if st.session_state.eval_history:
            st.divider()
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.eval_history = []
                st.rerun()

    # -------------------------------------------------------------------------
    # HERO BANNER
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🛡️ Credit Card Fraud Detection</div>
            <div class="hero-subtitle">
                Real-time financial anomaly detection platform powered by SMOTE-balanced 
                Machine Learning & PCA Vector Inferences.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 1: SINGLE TRANSACTION EVALUATION
    # -------------------------------------------------------------------------
    if mode == "💳 Single Transaction":
        st.markdown("### 💳 Transaction Input Parameters")

        # Basic Parameters Card
        col_t, col_a = st.columns(2)
        with col_t:
            time_val = st.number_input("Transaction Time Elapsed (Seconds)", value=0.0, step=1.0)
        with col_a:
            amount_val = st.number_input("Transaction Amount ($)", value=0.0, step=10.0)

        # Advanced PCA Feature Inputs
        with st.expander("⚙️ Advanced Anomaly Indicators (PCA Features V1 – V28)", expanded=False):
            st.caption("Adjust latent vector transformations derived from PCA decomposition.")
            v_inputs = []
            
            # Render grid of V1 to V28
            grid_cols = st.columns(4)
            for i in range(1, 29):
                with grid_cols[(i - 1) % 4]:
                    val = st.number_input(
                        f"V{i}",
                        value=0.0,
                        format="%.6f",
                        key=f"v_{i}"
                    )
                    v_inputs.append(val)

        predict_clicked = st.button("🔍 Analyze Transaction Risk", type="primary", use_container_width=True)

        if predict_clicked:
            # Prepare feature vector: Time + V1..V28 + Amount
            raw_data = [[time_val] + v_inputs + [amount_val]]
            columns_layout = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            df_input = pd.DataFrame(raw_data, columns=columns_layout)

            # Preprocessing & Inference
            X_scaled = scaler.transform(df_input)
            probability = float(model.predict_proba(X_scaled)[0][1])
            is_fraud = probability >= threshold

            # Save to history
            st.session_state.eval_history.append({
                "Time": time_val,
                "Amount": amount_val,
                "Probability": probability,
                "Status": "🚨 Fraud" if is_fraud else "✅ Genuine"
            })

            st.divider()

            # Results Section
            res_col1, res_col2 = st.columns([1, 1.2], gap="large")

            with res_col1:
                st.markdown("#### 📊 Risk Classification")
                if is_fraud:
                    st.markdown(f"""
                        <div class="result-card-fraud">
                            <h2 style="color: #f43f5e; margin: 0;">🚨 FRAUD DETECTED</h2>
                            <p style="color: #fca5a5; margin-top: 5px;">High Anomaly Score Identified</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-card-genuine">
                            <h2 style="color: #10b981; margin: 0;">✅ GENUINE TRANSACTION</h2>
                            <p style="color: #6ee7b7; margin-top: 5px;">Normal Behavioral Profile</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f'<div class="stat-card"><div class="stat-value" style="color:{"#f43f5e" if is_fraud else "#10b981"};">{probability*100:.2f}%</div><div class="stat-label">Fraud Probability</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="stat-card"><div class="stat-value">{threshold*100:.0f}%</div><div class="stat-label">Decision Threshold</div></div>', unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### 🎯 Risk Probability Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "font": {"color": "#ffffff", "size": 24}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                        "bar": {"color": "#f43f5e" if is_fraud else "#10b981"},
                        "bgcolor": "rgba(22, 31, 48, 0.8)",
                        "borderwidth": 1,
                        "bordercolor": "rgba(255,255,255,0.1)",
                        "steps": [
                            {"range": [0, threshold * 100], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [threshold * 100, 100], "color": "rgba(244, 63, 94, 0.15)"}
                        ],
                        "threshold": {
                            "line": {"color": "#ffffff", "width": 3},
                            "thickness": 0.75,
                            "value": threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#ffffff"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

    # -------------------------------------------------------------------------
    # MODE 2: BATCH PREDICTION VIA CSV
    # -------------------------------------------------------------------------
    elif mode == "📂 Batch Prediction":
        st.markdown("### 📂 Batch File Processing")

        uploaded_file = st.file_uploader(
            "Upload CSV containing transaction dataset (Time, V1..V28, Amount):",
            type=["csv"]
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.markdown("#### 📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("⚡ Execute Batch Fraud Screening", type="primary", use_container_width=True):
                with st.spinner("Processing transactions through scaling and inference engine..."):
                    # Separate features if target 'Class' exists
                    X = df.drop(columns=["Class"]) if "Class" in df.columns else df.copy()

                    # Scale and infer probabilities
                    X_scaled = scaler.transform(X)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    predictions = (probabilities >= threshold).astype(int)

                    # Append outputs to DataFrame
                    df_res = df.copy()
                    df_res["Fraud Probability"] = (probabilities * 100).round(2)
                    df_res["Prediction"] = np.where(predictions == 1, "🚨 Fraud", "✅ Genuine")

                    frauds = int((predictions == 1).sum())
                    genuine = int((predictions == 0).sum())
                    total = len(df_res)

                    st.success("Batch Analysis Complete!")
                    st.balloons()

                    # Summary KPI Metrics
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(f'<div class="stat-card"><div class="stat-value">{total:,}</div><div class="stat-label">Total Evaluated</div></div>', unsafe_allow_html=True)
                    with k2:
                        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #f43f5e;">{frauds:,}</div><div class="stat-label">Fraud Flagged</div></div>', unsafe_allow_html=True)
                    with k3:
                        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #10b981;">{genuine:,}</div><div class="stat-label">Genuine</div></div>', unsafe_allow_html=True)
                    with k4:
                        st.markdown(f'<div class="stat-card"><div class="stat-value" style="color: #60a5fa;">{(frauds/total)*100:.2f}%</div><div class="stat-label">Fraud Rate</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Visual Analytics Row
                    v_col1, v_col2 = st.columns([1, 1.2])

                    with v_col1:
                        st.markdown("#### 📈 Distribution Summary")
                        chart_df = pd.DataFrame({
                            "Status": ["Genuine", "Fraud"],
                            "Count": [genuine, frauds]
                        })
                        fig_bar = px.bar(
                            chart_df,
                            x="Status",
                            y="Count",
                            color="Status",
                            color_discrete_map={"Genuine": "#10b981", "Fraud": "#f43f5e"},
                            template="plotly_dark",
                            text_auto=True
                        )
                        fig_bar.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with v_col2:
                        st.markdown("#### 💸 Fraud Distribution by Amount")
                        fig_scatter = px.scatter(
                            df_res,
                            x="Amount",
                            y="Fraud Probability",
                            color="Prediction",
                            color_discrete_map={"✅ Genuine": "#10b981", "🚨 Fraud": "#f43f5e"},
                            template="plotly_dark",
                            hover_data=["Time", "Amount"]
                        )
                        fig_scatter.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    # Filtered Output Table & Download
                    st.markdown("#### 📋 Detailed Predictions Table")
                    
                    show_only_fraud = st.checkbox("Filter and display ONLY Fraud Flagged Transactions")
                    filtered_df = df_res[df_res["Prediction"] == "🚨 Fraud"] if show_only_fraud else df_res

                    st.dataframe(
                        filtered_df[["Time", "Amount", "Fraud Probability", "Prediction"]],
                        use_container_width=True
                    )

                    # Export Results CSV
                    csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Complete Predictions CSV",
                        data=csv_bytes,
                        file_name="fraud_predictions_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # -------------------------------------------------------------------------
    # HISTORICAL EVALUATIONS DRAWER
    # -------------------------------------------------------------------------
    if st.session_state.eval_history:
        st.divider()
        st.markdown("### 📜 Session Inferences History")
        st.dataframe(pd.DataFrame(st.session_state.eval_history), use_container_width=True)

    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="footer-container">
            Credit Card Fraud Detection Engine • Machine Learning Powered Risk Assessment<br>
            <small style="color: #4b5563;">Logistic Regression & SMOTE Balanced Sampling Pipeline</small><br>
            <small style="color: #6b7280;">© 2026 | Developed by <b>Madhur</b></small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
