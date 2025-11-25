import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from prophet.plot import plot_plotly
import os

st.set_page_config(page_title="Colgate Sales Forecasting", layout="wide")

st.title("📈 Colgate Oral-Care: Demand Forecasting System")
st.markdown("Forecast weekly demand for Toothpaste, Toothbrush & Mouthwash.")

# -------------------------------------------------------------------------
# PATH FIX (works everywhere)
# -------------------------------------------------------------------------

# Get directory of the current file (app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "../data/colgate_sales_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "../models")

# Load data
df = pd.read_csv(DATA_PATH)
products = df["product"].unique()

# Sidebar product dropdown
option = st.sidebar.selectbox("Select Product", products)

# Load appropriate model
MODEL_PATH = os.path.join(MODELS_DIR, f"{option}_prophet_model.pkl")
model = joblib.load(MODEL_PATH)

# Prepare data
temp = df[df["product"] == option].rename(
    columns={"date": "ds", "weekly_sales": "y"}
)

# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
st.subheader(f"📊 Historical Sales — {option}")
fig1 = px.line(temp, x="ds", y="y", title=f"{option} Weekly Sales (Historical)")
st.plotly_chart(fig1, use_container_width=True)

# Forecast
future = model.make_future_dataframe(periods=20, freq="W")
forecast = model.predict(future)

st.subheader("🔮 Forecast for Next 20 Weeks")
fig2 = plot_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🧾 Forecast Data (Next 20 Weeks)")
st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20)
)
