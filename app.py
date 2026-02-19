import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- Load Model ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

MAE = 74

# ---------------- Styling ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
h1, h2, h3 {
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("📊 Customer Lifetime Value (CLV) Prediction System")

tab_overview, tab_prediction, tab_analytics = st.tabs(
    ["📘 Overview", "🔮 Prediction", "📈 Analytics & Insights"]
)

# ================= OVERVIEW =================
with tab_overview:

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📌 Project Overview")
        st.write(
            "Customer Lifetime Value (CLV) estimates the total revenue a business "
            "can expect from a customer throughout their relationship."
        )

        st.markdown("### 🔑 Key Features")
        st.write("""
        • ML-based CLV prediction  
        • Feature impact explanation  
        • Business recommendations  
        • Visual analytics  
        """)

        st.markdown("### 🏢 Business Use Cases")
        st.write("""
        • Customer segmentation  
        • Targeted marketing campaigns  
        • Retention strategy planning  
        • Revenue forecasting  
        """)

    with col2:
        st.image(
            "https://miro.medium.com/v2/resize:fit:720/format:webp/1*g1WtmMn6n0yz-W_bh0JpWQ.jpeg",
            use_container_width=True
        )

# ================= PREDICTION =================
with tab_prediction:

    st.subheader("🔢 Enter Customer Details")

    recency = st.number_input("Recency (Days since last purchase)", min_value=0)
    num_invoices = st.number_input("Number of Invoices", min_value=0)
    spent_amount = st.number_input("Total Spent Amount (₹)", min_value=0.0, step=100.0)

    st.markdown("---")

    if st.button("🚀 Generate CLV Report"):

        features = np.array([[recency, num_invoices, spent_amount]])
        prediction = model.predict(features)[0]

        st.subheader("📊 Customer Lifetime Value Report")
        st.markdown(
            f"<h1 style='color:#1f77b4;'>₹ {prediction:,.0f}</h1>",
            unsafe_allow_html=True
        )

        confidence = round(max(0, 100 - (MAE / max(prediction, 1)) * 100), 2)
        st.progress(int(confidence))

        st.markdown("---")

        st.subheader("🔍 Feature Impact Explanation")

        if spent_amount > 20000:
            st.write("• High spending significantly increased CLV.")
        elif spent_amount > 8000:
            st.write("• Moderate spending positively supported CLV.")
        else:
            st.write("• Low spending reduced CLV.")

        if num_invoices > 10:
            st.write("• Frequent purchases boosted CLV.")

        if recency < 30:
            st.write("• Recent engagement improved CLV.")

        # -------- Feature Contribution Chart --------
        st.subheader("📊 Feature Contribution to CLV")

        df = pd.DataFrame({
            "Feature": ["Monetary", "Frequency", "Recency"],
            "Impact Score": [
                spent_amount * 0.6 / 100,
                num_invoices * 0.3,
                (100 - recency) * 0.1
            ]
        })

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(df["Feature"], df["Impact Score"])
        ax.set_title("RFM Contribution to CLV", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)

# ================= ANALYTICS =================
with tab_analytics:

    st.subheader("📈 Business Insights & Visual Analysis")

    col_a, col_b = st.columns(2)

    # -------- CLV Comparison --------
    with col_a:
        st.markdown("### 💰 CLV Comparison")

        comparison_df = pd.DataFrame({
            "Category": [
                "Predicted CLV",
                "Average Customer CLV",
                "High Value Benchmark"
            ],
            "CLV Value": [12000, 8000, 18000]
        })

        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.bar(comparison_df["Category"], comparison_df["CLV Value"])
        ax1.set_ylabel("CLV (₹)")
        ax1.set_title("Customer CLV Comparison", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig1)

    # -------- Customer Segmentation --------
    with col_b:
        st.markdown("### 👥 Customer Segment Distribution")

        segment_df = pd.DataFrame({
            "Segment": ["Low Value", "Medium Value", "High Value"],
            "Customers": [50, 35, 15]
        })

        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        ax2.pie(
            segment_df["Customers"],
            labels=segment_df["Segment"],
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={"edgecolor": "white"}
        )
        ax2.set_title("Customer Segmentation by CLV", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")

    # -------- Customer Behavior --------
    st.markdown("### 📊 Customer Behavior Analysis")

    behavior_df = pd.DataFrame({
        "Invoices": [2, 5, 8, 12, 18],
        "Spending": [2000, 6000, 9000, 15000, 25000]
    })

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    ax3.scatter(behavior_df["Invoices"], behavior_df["Spending"])
    ax3.set_xlabel("Number of Invoices")
    ax3.set_ylabel("Total Spent (₹)")
    ax3.set_title("Spending vs Purchase Frequency", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig3)
