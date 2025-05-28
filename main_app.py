import streamlit as st
import time
from datetime import datetime, timezone
import pytz
from config import COMISION, CAPITAL_INICIAL
from datos import obtener_datos
from estrategia import generar_señales_historicas
from estrategia import calcular_senal_actual
from backtesting import evaluar_estrategia
from simulacion import comprar, vender
from visualizaciones import graficar_precio, graficar_ema, graficar_rsi, graficar_macd
from visualizaciones import graficar_backtest_entradas_salidas

# Configuración inicial de página
st.set_page_config(page_title="Bot RSI BTC + MACD + EMA", layout="wide")

# Inicializar variables de sesión
for key, val in {"modo": "Prueba", "usdt": CAPITAL_INICIAL, "btc": 0.0, "historial": [], "last_update": 0.0}.items():
    if key not in st.session_state:
        st.session_state[key] = val

symbol = st.sidebar.selectbox(
    "Selecciona la criptomoneda",
    [
        "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD",
        "MATIC-USD", "LTC-USD", "BNB-USD", "DOT-USD", "SHIB-USD", "AVAX-USD",
        "UNI-USD", "LINK-USD", "ATOM-USD", "FIL-USD", "AAVE-USD", "NEAR-USD"
    ],
    index=0
)
period = st.sidebar.selectbox("Periodo", ["1d", "3d", "7d", "14d"], index=1)
interval = "5m"

st.title(f"🤖 Bot de Trading {symbol} con RSI, MACD y EMA")

import time

REFRESH_INTERVAL = 30  # segundos

# Botón manual
if st.button("🔁 Refrescar ahora"):
    st.session_state["last_update"] = time.time()

# Auto-refresco cada N segundos
if time.time() - st.session_state["last_update"] > REFRESH_INTERVAL:
    st.session_state["last_update"] = time.time()
    st.rerun()


# Mostrar hora local y UTC
local_tz = pytz.timezone("America/Mexico_City")
st.write(f"🕒 Hora local: {datetime.now(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.write(f"🕒 Hora UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}")

tabs = st.tabs(["📊 Análisis en Tiempo Real", "📈 Backtesting"])

with tabs[0]:
    df = obtener_datos(symbol, interval, period)

    if not df.empty:
        precio = df['Close'].iloc[-1]

        # Señal combinada simplificada (resumen)
        senal, s_rsi, s_ema, s_macd, rsi, ema_rapida, ema_lenta, macd, macd_signal = calcular_senal_actual(df)

        st.subheader("🔍 Señal técnica actual")
        st.info(senal)

        col1, col2, col3 = st.columns(3)

        # Columna 1 - RSI
        with col1:
            st.metric("📉 RSI", f"{rsi:.2f}", help=f"Señal RSI: {s_rsi}")

        # Columna 2 - EMAs (formato precio)
        with col2:
            st.metric("📊 EMA Rápida (12)", f"${ema_rapida:,.2f}")
            st.metric("📊 EMA Lenta (26)", f"${ema_lenta:,.2f}", help=f"Señal EMA: {s_ema}")

        # Columna 3 - MACD y Señal (también como precios si deseas)
        with col3:
            st.metric("📈 MACD", f"{macd:,.2f}")
            st.metric("📈 Señal MACD", f"{macd_signal:,.2f}", help=f"Señal MACD: {s_macd}")



        # Mostrar gráficos
        st.subheader("📈 Gráficos")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(graficar_precio(df, symbol), use_container_width=True)
            st.plotly_chart(graficar_ema(df), use_container_width=True)
        with col_g2:
            st.plotly_chart(graficar_rsi(df), use_container_width=True)
            st.plotly_chart(graficar_macd(df), use_container_width=True)

        # Portafolio ficticio
        st.subheader("💼 Portafolio ficticio")
        st.metric("💵 USDT", f"${st.session_state['usdt']:.2f}")
        st.metric(f"🪙 {symbol.split('-')[0]}", f"{st.session_state['btc']:.6f}")

        col_buy, col_sell = st.columns(2)
        with col_buy:
            usdt_para_comprar = st.number_input("USDT para comprar", min_value=0.0, max_value=st.session_state['usdt'], step=10.0)
            if st.button("Comprar"):
                st.session_state['usdt'], btc, ok = comprar(st.session_state['usdt'], precio, usdt_para_comprar, COMISION)
                if ok:
                    st.session_state['btc'] += btc
        with col_sell:
            btc_para_vender = st.number_input("BTC a vender", min_value=0.0, max_value=st.session_state['btc'], step=0.0001)
            if st.button("Vender"):
                st.session_state['btc'], usdt, ok = vender(st.session_state['btc'], precio, btc_para_vender, COMISION)
                if ok:
                    st.session_state['usdt'] += usdt

with tabs[1]:
    st.subheader("📈 Backtesting")
    rsi_p = 14
    ema_f = 12
    ema_l = 26
    macd_f = 12
    macd_s = 26
    macd_sig = 9

    df = obtener_datos(symbol, interval, period)
    df.dropna(inplace=True)
    df['Señal'] = generar_señales_historicas(df)
    capital_final = evaluar_estrategia(df)

    st.metric("💰 Capital final simulado", f"${capital_final:.2f}")
    df_tabla = df.copy()
    for col in ["Close", "EMA_rapida", "EMA_lenta"]:
        df_tabla[col] = df_tabla[col].apply(lambda x: f"${x:,.2f}")
    df_tabla["MACD"] = df_tabla["MACD"].apply(lambda x: f"{x:,.2f}")
    df_tabla["MACD_Signal"] = df_tabla["MACD_Signal"].apply(lambda x: f"{x:,.2f}")
    df_tabla["RSI"] = df_tabla["RSI"].apply(lambda x: f"{x:.2f}")

    st.dataframe(df_tabla[["Close", "RSI", "EMA_rapida", "EMA_lenta", "MACD", "MACD_Signal", "Señal"]].tail(100))


    st.plotly_chart(graficar_backtest_entradas_salidas(df, symbol), use_container_width=True)

