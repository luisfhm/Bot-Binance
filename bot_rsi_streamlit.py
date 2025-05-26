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
st.set_page_config(page_title="Bot RSI BTC + MACD + EMA", layout="wide")

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
period = "7d"
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

# Función para calcular EMA
def calcular_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Función para calcular MACD
def calcular_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calcular_ema(data, fast)
    ema_slow = calcular_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calcular_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# Función para obtener datos y calcular indicadores
def obtener_datos(period):
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df['RSI'] = calcular_rsi(df['Close'], 14)
    df['EMA_12'] = calcular_ema(df['Close'], 12)
    df['EMA_26'] = calcular_ema(df['Close'], 26)
    macd_line, signal_line, hist = calcular_macd(df['Close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    return df

period = st.sidebar.selectbox(
    "Selecciona el intervalo de tiempo para la gráfica",
    options=["1d", "3d", "7d", "14d", "1mo", "3mo", "6mo"],
    index=2,  # por defecto 7d
)


# Obtener y mostrar datos
df = obtener_datos(period)
df.columns = df.columns.get_level_values(0)

st.title("🤖 Bot de Trading BTC/USDT con RSI, MACD y EMA")

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
        ema_12 = float(df['EMA_12'].dropna().iloc[-1])
        ema_26 = float(df['EMA_26'].dropna().iloc[-1])
        macd = float(df['MACD'].dropna().iloc[-1])
        macd_signal = float(df['MACD_Signal'].dropna().iloc[-1])

        # --- Señal combinada de compra/venta ---
        # RSI
        señal_rsi = 0
        if rsi < 30:
            señal_rsi = 1  # compra
        elif rsi > 70:
            señal_rsi = -1  # venta

        # EMA crossover (últimos dos puntos para detectar cruce)
        ema12_ayer = df['EMA_12'].dropna().iloc[-2] if len(df['EMA_12'].dropna()) > 1 else ema_12
        ema26_ayer = df['EMA_26'].dropna().iloc[-2] if len(df['EMA_26'].dropna()) > 1 else ema_26
        señal_ema = 0
        if (ema12_ayer < ema26_ayer) and (ema_12 > ema_26):
            señal_ema = 1  # cruce al alza: compra
        elif (ema12_ayer > ema26_ayer) and (ema_12 < ema_26):
            señal_ema = -1  # cruce a la baja: venta

        # MACD crossover (últimos dos puntos)
        macd_ayer = df['MACD'].dropna().iloc[-2] if len(df['MACD'].dropna()) > 1 else macd
        macd_signal_ayer = df['MACD_Signal'].dropna().iloc[-2] if len(df['MACD_Signal'].dropna()) > 1 else macd_signal
        señal_macd = 0
        if (macd_ayer < macd_signal_ayer) and (macd > macd_signal):
            señal_macd = 1  # cruce al alza: compra
        elif (macd_ayer > macd_signal_ayer) and (macd < macd_signal):
            señal_macd = -1  # cruce a la baja: venta

        # Suma señales
        suma_señales = señal_rsi + señal_ema + señal_macd

        if suma_señales > 0:
            señal_final = "🟢 Señal combinada de COMPRA"
        elif suma_señales < 0:
            señal_final = "🔴 Señal combinada de VENTA"
        else:
            señal_final = "⚪ Sin señal clara"

        st.info(señal_final)

        # --- FIN señal ---

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

        # Gráficos con Plotly
        st.subheader("📈 Gráficos de Precio, RSI, MACD y EMAs")
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

            fig_ema = px.line(df, x=df.index, y=['EMA_12', 'EMA_26'], title='📊 Medias Móviles Exponenciales (EMA)')
            fig_ema.update_layout(template="plotly_white")
            st.plotly_chart(fig_ema, use_container_width=True)

        with col2:
            fig_rsi = px.line(df, x=df.index, y='RSI', title='📉 RSI de BTC')
            fig_rsi.update_layout(
                xaxis_title="Fecha",
                yaxis_title="RSI",
                template="plotly_white"
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

            fig_macd = px.line(df, x=df.index, y=['MACD', 'MACD_Signal'], title='📈 MACD y Señal')
            fig_macd.update_layout(template="plotly_white")
            st.plotly_chart(fig_macd, use_container_width=True)

        # Métricas actuales
        st.metric("💰 Precio actual BTC/USDT", f"${precio:,.2f}")
        st.metric("📉 RSI actual", f"{rsi:.2f}")
        st.metric("📊 EMA 12 actual", f"${ema_12:,.2f}")
        st.metric("📊 EMA 26 actual", f"${ema_26:,.2f}")
        st.metric("📈 MACD actual", f"{macd:.4f}")
        st.metric("📈 MACD Señal actual", f"{macd_signal:.4f}")

        # Sección de portafolio en modo prueba
        if modo == "Prueba":
            st.subheader("💼 Portafolio ficticio")
            st.metric("💵 USDT disponible", f"${st.session_state['usdt']:,.2f}")
            st.metric("🪙 BTC en cartera", f"{st.session_state['btc']:.6f}")
            valor_total = st.session_state['usdt'] + st.session_state['btc'] * precio
            st.metric("📊 Valor total del portafolio", f"${valor_total:,.2f}")

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
                            st.session_state['usdt'] -= costo_con_comision
                            st.session_state['btc'] += btc_comprado
                            st.session_state['historial'].append({
                                'tipo': 'Compra',
                                'hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'precio': round(precio, 2),
                                'btc': round(btc_comprado, 6),
                                'usdt': -round(costo_con_comision, 2)
                            })
                            st.success(f"Compra simulada: BTC comprado {btc_comprado:.6f}, USDT gastado neto {costo_con_comision:.2f}")
                        else:
                            st.warning("No tienes suficiente USDT para cubrir la comisión.")
                    else:
                        st.warning("Ingresa una cantidad válida para comprar.")

            with col_sell:
                btc_para_vender = st.number_input(
                    "Cantidad BTC a vender",
                    min_value=0.0, max_value=st.session_state['btc'], step=0.0001, format="%.6f"
                )
                if st.button("Vender BTC"):
                    if btc_para_vender > 0 and btc_para_vender <= st.session_state['btc']:
                        ingreso_usdt = (btc_para_vender * precio) * (1 - COMISION)
                        st.session_state['btc'] -= btc_para_vender
                        st.session_state['usdt'] += ingreso_usdt
                        st.session_state['historial'].append({
                            'tipo': 'Venta',
                            'hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'precio': round(precio, 2),
                            'btc': -round(btc_para_vender, 6),
                            'usdt': round(ingreso_usdt, 2)
                        })
                        st.success(f"Venta simulada: BTC vendido {btc_para_vender:.6f}, USDT recibido neto {ingreso_usdt:.2f}")
                    else:
                        st.warning("Ingresa una cantidad válida para vender.")

            # Mostrar historial
            if st.session_state['historial']:
                st.write("### 📝 Historial de operaciones")
                df_hist = pd.DataFrame(st.session_state['historial'])
                st.dataframe(df_hist)

        else:
            st.info("Modo Real no implementado en este demo.")

    except Exception as e:
        st.error(f"Error al procesar datos: {e}")

else:
    st.warning("No se pudieron obtener datos de BTC.")