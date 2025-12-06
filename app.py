import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="Dashboard Climático", layout="wide")

# ------------------------------
# CARGA DE DATOS
# ------------------------------
@st.cache_data
def cargar_datos():
    df =  pd.read_csv("clima_peru.csv", parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df

clima = cargar_datos()

clima["year"] = clima["date"].dt.year
clima["month"] = clima["date"].dt.month
clima["day"] = clima["date"].dt.day
clima["dayofweek"] = clima["date"].dt.dayofweek
clima["week"] = clima["date"].dt.isocalendar().week.astype(int)

# ------------------------------
# DATASET MENSUAL
# ------------------------------
df_mensual = (
    clima
    .groupby(['city', 'year', 'month'], as_index=False)
    .agg({
        'temperature_2m_mean': 'mean',
        'precipitation_sum': 'mean',
        'relative_humidity_2m_mean': 'mean',
        'wind_speed_10m_mean': 'mean'
    })
)

df_mensual['fecha'] = pd.to_datetime(
    df_mensual['year'].astype(str) + '-' +
    df_mensual['month'].astype(str) + '-01'
)

# ------------------------------
# FUNCION FORECAST
# ------------------------------
def forecast_ciudad(ciudad, pasos=6):

    df_c = df_mensual[df_mensual['city'] == ciudad].copy()
    df_c = df_c.set_index('fecha')
    y = df_c['temperature_2m_mean']

    modelo = SARIMAX(
        y,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    ).fit(disp=False)

    forecast = modelo.get_forecast(steps=pasos)
    pred = forecast.predicted_mean
    ic = forecast.conf_int()

    return y, pred, ic

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("Panel de Control")
ciudad = st.sidebar.selectbox(
    "Selecciona una ciudad:",
    sorted(df_mensual['city'].unique())
)

# ------------------------------
# TITULO
# ------------------------------
st.title(f"Dashboard Climático – {ciudad}")

df_city = df_mensual[df_mensual['city'] == ciudad]

# ------------------------------
# GRAFICOS PRINCIPALES (4 INDICADORES)
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_city['fecha'],
        y=df_city['temperature_2m_mean'],
        mode='lines+markers',
        name='Temperatura'
    ))
    fig1.update_layout(title="Temperatura Media (°C)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_city['fecha'],
        y=df_city['precipitation_sum'],
        mode='lines+markers',
        name='Precipitación'
    ))
    fig2.update_layout(title="Precipitación")
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_city['fecha'],
        y=df_city['relative_humidity_2m_mean'],
        mode='lines+markers',
        name='Humedad'
    ))
    fig3.update_layout(title="Humedad Relativa")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_city['fecha'],
        y=df_city['wind_speed_10m_mean'],
        mode='lines+markers',
        name='Viento'
    ))
    fig4.update_layout(title="Velocidad del Viento")
    st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# FORECAST
# ------------------------------
st.subheader("Forecast de Temperatura Media (6 meses)")

y_real, y_pred, ic = forecast_ciudad(ciudad, pasos=6)

fig_f = go.Figure()

fig_f.add_trace(go.Scatter(
    x=y_real.index,
    y=y_real,
    name="Serie Real"
))

fig_f.add_trace(go.Scatter(
    x=y_pred.index,
    y=y_pred,
    name="Forecast",
    line=dict(dash="dash")
))

fig_f.add_trace(go.Scatter(
    x=ic.index,
    y=ic.iloc[:,1],
    line=dict(width=0),
    showlegend=False
))

fig_f.add_trace(go.Scatter(
    x=ic.index,
    y=ic.iloc[:,0],
    fill="tonexty",
    name="Intervalo de Confianza"
))

st.plotly_chart(fig_f, use_container_width=True)
