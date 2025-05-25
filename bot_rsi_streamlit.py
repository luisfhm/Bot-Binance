import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import time
from datetime import datetime, timezone
import warnings
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración inicial
st.set_page_config(page_title="Bot RSI BTC", layout="wide")

# Variables de sesión
if 'modo' not in st.session_state:
    st.session_state['modo'] = 'Prueba'
if 'usdt' not in st.session_state:
    st.session_state['usdt'] = 1000.0
if 'btc' not in st.session_state:
    st.session_state['btc'] = 0.0
if 'historial' not in st.session_state:
    st.session_state['historial'] = []
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = 0.0

# Parámetros y comisión
symbol = "BTC-USD"
interval = "5m"
period = "1d"
COMISION = 0.001  # 0.1% comisión

# Función para calcular RSI
def calcular_rsi(data, window):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función para obtener datos y calcular RSI
def obtener_datos():
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df['RSI'] = calcular_rsi(df['Close'], 14)
    return df

# Interfaz
st.title("🤖 Bot de Trading BTC/USDT con RSI")

modo = st.sidebar.selectbox("🛠 Modo de operación", ["Prueba", "Real"], key="modo_select")
st.session_state['modo'] = modo
st.info(f"🧪 Estás en **modo {modo}**")

# Botón de refresco manual
if st.button("🔁 Refrescar ahora"):
    st.session_state['last_update'] = time.time()

# Refresco automático cada 60 segundos
REFRESH_INTERVAL = 60
if time.time() - st.session_state['last_update'] > REFRESH_INTERVAL:
    st.session_state['last_update'] = time.time()

# Obtener y mostrar datos
df = obtener_datos()
df.columns = df.columns.get_level_values(0)

# Obtener hora local
local_tz = pytz.timezone("America/Mexico_City")  # Cambia a tu zona horaria real
hora_local = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
hora_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

st.write(f"🕒 Hora actual en UTC: {hora_utc}")
st.write(f"🕒 Hora local actual: {hora_local}")

if not df.empty and 'Close' in df.columns and 'RSI' in df.columns:
    try:
        precio = float(df['Close'].dropna().iloc[-1])
        rsi = float(df['RSI'].dropna().iloc[-1])

        # Gráficos con Plotly
        st.subheader("📈 Gráfico de precio y RSI")
        col1, col2 = st.columns(2)

        with col1:
            fig_close = px.line(df, x=df.index, y='Close', title='💰 Precio BTC/USDT')
            fig_close.update_layout(
                yaxis_tickprefix="$",
                yaxis_tickformat=",",
                xaxis_title="Fecha",
                yaxis_title="Precio (USDT)",
                template="plotly_white"
            )
            st.plotly_chart(fig_close, use_container_width=True)

        with col2:
            fig_rsi = px.line(df, x=df.index, y='RSI', title='📉 RSI de BTC')
            fig_rsi.update_layout(
                xaxis_title="Fecha",
                yaxis_title="RSI",
                template="plotly_white"
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        # Métricas actuales
        st.metric("💰 Precio actual BTC/USDT", f"${precio:,.2f}")
        st.metric("📉 RSI actual", f"{rsi:.2f}")

    except IndexError:
        st.warning("No se pudieron obtener los últimos valores de precio o RSI.")
else:
    st.warning("No se pudieron obtener datos válidos.")

# Sección de portafolio en modo prueba
if modo == "Prueba":
    st.subheader("💼 Portafolio ficticio")
    st.metric("💵 USDT disponible", f"${st.session_state['usdt']:,.2f}")
    st.metric("🪙 BTC en cartera", f"{st.session_state['btc']:.6f}")
    valor_total = st.session_state['usdt'] + st.session_state['btc'] * precio
    st.metric("📊 Valor total del portafolio", f"${valor_total:,.2f}")

    # Señal RSI
    señal = ""
    if rsi < 30:
        señal = "🟢 Señal de compra (RSI bajo)"
    elif rsi > 70:
        señal = "🔴 Señal de venta (RSI alto)"
    else:
        señal = "⚪ Sin señal clara"
    st.info(señal)

    st.write("### Operar BTC/USDT")

    # Entradas para comprar y vender
    col_buy, col_sell = st.columns(2)

    with col_buy:
        usdt_para_comprar = st.number_input(
            "Cantidad USDT a usar para comprar BTC",
            min_value=0.0, max_value=st.session_state['usdt'], step=10.0, format="%.2f"
        )
        if st.button("Comprar BTC"):
            if usdt_para_comprar > 0 and usdt_para_comprar <= st.session_state['usdt']:
                costo_con_comision = usdt_para_comprar * (1 + COMISION)
                if costo_con_comision <= st.session_state['usdt']:
                    btc_comprado = (usdt_para_comprar / precio) * (1 - COMISION)
                    st.session_state['btc'] += btc_comprado
                    st.session_state['usdt'] -= costo_con_comision
                    st.session_state['historial'].append({
                        'tipo': 'Compra',
                        'hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'precio': round(precio, 2),
                        'btc': round(btc_comprado, 6),
                        'usdt': -round(costo_con_comision, 2)
                    })
                    st.success(f"Compra simulada: BTC comprado {btc_comprado:.6f}, USDT gastado con comisión {costo_con_comision:.2f}")
                else:
                    st.warning("No tienes suficiente USDT para cubrir la compra con comisión.")
            else:
                st.warning("Ingresa una cantidad válida para comprar.")

    with col_sell:
        btc_para_vender = st.number_input(
            "Cantidad BTC a vender",
            min_value=0.0, max_value=st.session_state['btc'], step=0.0001, format="%.6f"
        )
        if st.button("Vender BTC"):
            if btc_para_vender > 0 and btc_para_vender <= st.session_state['btc']:
                usdt_obtenido = btc_para_vender * precio * (1 - COMISION)
                st.session_state['btc'] -= btc_para_vender
                st.session_state['usdt'] += usdt_obtenido
                st.session_state['historial'].append({
                    'tipo': 'Venta',
                    'hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'precio': round(precio, 2),
                    'btc': -round(btc_para_vender, 6),
                    'usdt': round(usdt_obtenido, 2)
                })
                st.success(f"Venta simulada: BTC vendido {btc_para_vender:.6f}, USDT recibido con comisión {usdt_obtenido:.2f}")
            else:
                st.warning("Ingresa una cantidad válida para vender.")

    # Historial formateado
    if st.session_state['historial']:
        st.subheader("📜 Historial de operaciones")
        historial_df = pd.DataFrame(st.session_state['historial'])
        historial_df['precio'] = historial_df['precio'].map(lambda x: f"${x:,.2f}")
        historial_df['usdt'] = historial_df['usdt'].map(lambda x: f"${x:,.2f}")
        historial_df['btc'] = historial_df['btc'].map(lambda x: f"{x:.6f}")
        st.table(historial_df[['hora', 'tipo', 'precio', 'btc', 'usdt']])
