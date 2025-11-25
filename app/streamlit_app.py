import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from prophet.plot import plot_plotly

st.set_page_config(page_title="Colgate Sales Forecasting", layout="wide")

st.title("📈 Colgate Oral-Care: Demand Forecasting System")
st.markdown("Forecast weekly demand for Toothpaste, Toothbrush & Mouthwash.")

# Load Data
df = pd.read_csv("colgate_sales_data.csv")
products = df["product"].unique()

# Sidebar
option = st.sidebar.selectbox("Select Product", products)

# Load Model
model = joblib.load(f"{option}_prophet_model.pkl")

temp = df[df["product"] == option].rename(columns={"date": "ds", "weekly_sales": "y"})

st.subheader(f"📊 Historical Sales — {option}")
fig1 = px.line(temp, x="ds", y="y", title=f"{option} Weekly Sales")
st.plotly_chart(fig1, use_container_width=True)

# Forecast
future = model.make_future_dataframe(periods=20, freq="W")
forecast = model.predict(future)

st.subheader("🔮 Forecast for Next 20 Weeks")
fig2 = plot_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🧾 Forecast Data")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20))
