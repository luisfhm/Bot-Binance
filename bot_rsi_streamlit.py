import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import time
from datetime import datetime, timezone
import warnings
import pytz
import plotly.graph_objects as go
import ta

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
symbol = st.sidebar.selectbox(
    "Selecciona la criptomoneda",
    options=["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD"],
    index=0
)

capital_inicial = st.sidebar.number_input(
    "💰 Capital inicial ($)", min_value=100.0, max_value=100000.0, value=1000.0, step=100.0
)
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
def obtener_datos(symbol, period, rsi_periodo, ema_rapida, ema_lenta, macd_fast, macd_slow, macd_signal):
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df['RSI'] = calcular_rsi(df['Close'], rsi_periodo)
    df['EMA_rapida'] = calcular_ema(df['Close'], ema_rapida)
    df['EMA_lenta'] = calcular_ema(df['Close'], ema_lenta)
    macd_line, signal_line, hist = calcular_macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    return df

def evaluar_parametros(symbol, period, rsi_p, ema_f, ema_l, macd_f, macd_s, macd_sig):
    df = obtener_datos(symbol, period, rsi_p, ema_f, ema_l, macd_f, macd_s, macd_sig)
    df = df.dropna().copy()
    df["Señal"] = generar_señales_historicas(df)

    capital = 1000
    btc = 0

    for i in range(len(df)):
        señal = df["Señal"].iloc[i]
        precio = df["Close"].iloc[i]
        if señal == "COMPRA" and capital > 0:
            btc = (capital / precio) * (1 - COMISION)
            capital = 0
        elif señal == "VENTA" and btc > 0:
            capital = (btc * precio) * (1 - COMISION)
            btc = 0

    return capital + btc * df["Close"].iloc[-1]



period = st.sidebar.selectbox(
    "Selecciona el intervalo de tiempo para la gráfica",
    options=["1d", "3d", "7d", "14d", "1mo", "3mo", "6mo"],
    index=1,  # por defecto 3d
)

st.title(f"🤖 Bot de Trading {symbol} con RSI, MACD y EMA")

# Obtener hora local
local_tz = pytz.timezone("America/Mexico_City")  # Cambia a tu zona horaria real
hora_local = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
hora_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

st.write(f"🕒 Hora actual en UTC: {hora_utc}")
st.write(f"🕒 Hora local actual: {hora_local}")

tabs = st.tabs(["📊 Análisis en Tiempo Real", "📈 Análisis Histórico"])


with tabs[0]:
    st.header("Ajuste de parámetros técnicos")
    
    col1, col2 = st.columns(2)

    with col1:
        rsi_periodo = st.slider("RSI Periodo", 5, 30, 14, 1)
        ema_rapida = st.slider("EMA Rápida", 5, 50, 12, 1)
        macd_fast = st.slider("MACD Fast", 5, 50, 12, 1)

    with col2:
        ema_lenta = st.slider("EMA Lenta", 10, 100, 26, 1)
        macd_slow = st.slider("MACD Slow", 10, 100, 26, 1)
        macd_signal = st.slider("MACD Signal", 5, 30, 9, 1)

    st.markdown("---")

    # Obtener y mostrar datos
    df = obtener_datos(symbol, period, rsi_periodo, ema_rapida, ema_lenta, macd_fast, macd_slow, macd_signal)
    df.columns = df.columns.get_level_values(0)

    if not df.empty and 'Close' in df.columns and 'RSI' in df.columns:
        try:
            precio = float(df['Close'].dropna().iloc[-1])
            rsi = float(df['RSI'].dropna().iloc[-1])
            ema_12 = float(df['EMA_rapida'].dropna().iloc[-1])
            ema_26 = float(df['EMA_lenta'].dropna().iloc[-1])
            macd = float(df['MACD'].dropna().iloc[-1])
            macd_signal_val = float(df['MACD_Signal'].dropna().iloc[-1])

            # --- Señal combinada ---
            señal_rsi = 0
            if rsi < 30:
                señal_rsi = 1
            elif rsi > 70:
                señal_rsi = -1

            ema12_ayer = df['EMA_rapida'].dropna().iloc[-2]
            ema26_ayer = df['EMA_lenta'].dropna().iloc[-2]
            señal_ema = 0
            if (ema12_ayer < ema26_ayer) and (ema_12 > ema_26):
                señal_ema = 1
            elif (ema12_ayer > ema26_ayer) and (ema_12 < ema_26):
                señal_ema = -1

            macd_ayer = df['MACD'].dropna().iloc[-2] if len(df['MACD'].dropna()) > 1 else macd
            macd_signal_ayer = df['MACD_Signal'].dropna().iloc[-2] if len(df['MACD_Signal'].dropna()) > 1 else macd_signal_val
            señal_macd = 0
            if (macd_ayer < macd_signal_ayer) and (macd > macd_signal_val):
                señal_macd = 1
            elif (macd_ayer > macd_signal_ayer) and (macd < macd_signal_val):
                señal_macd = -1

            suma_señales = señal_rsi + señal_ema + señal_macd

            if suma_señales > 0:
                señal_final = "🟢 Señal combinada de COMPRA"
            elif suma_señales < 0:
                señal_final = "🔴 Señal combinada de VENTA"
            else:
                señal_final = "⚪ Sin señal clara"

            st.info(señal_final)

            # Modo operación sidebar
            modo = st.sidebar.selectbox("🛠 Modo de operación", ["Prueba", "Real"], key="modo_select")
            st.session_state['modo'] = modo
            st.info(f"🧪 Estás en **modo {modo}**")

            # Botón refresco manual y refresco automático
            if st.button("🔁 Refrescar ahora"):
                st.session_state['last_update'] = time.time()

            REFRESH_INTERVAL = 30
            if time.time() - st.session_state.get('last_update', 0) > REFRESH_INTERVAL:
                st.session_state['last_update'] = time.time()

            st.markdown("---")
            st.subheader("📈 Gráficos de Precio, RSI, MACD y EMAs")

            col_graf1, col_graf2 = st.columns(2)
            with col_graf1:
                fig_close = px.line(df, x=df.index, y='Close', title=f'💰 Precio {symbol}')
                fig_close.update_layout(
                    yaxis_tickprefix="$",
                    yaxis_tickformat=",",
                    xaxis_title="Fecha",
                    yaxis_title="Precio (USDT)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_close, use_container_width=True)

                fig_ema = px.line(df, x=df.index, y=['EMA_rapida', 'EMA_lenta'], title='📊 EMAs personalizadas')
                fig_ema.update_layout(template="plotly_white")
                st.plotly_chart(fig_ema, use_container_width=True)

            with col_graf2:
                fig_rsi = px.line(df, x=df.index, y='RSI', title=f'📉 RSI de {symbol}')
                fig_rsi.update_layout(xaxis_title="Fecha", yaxis_title="RSI", template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)

                fig_macd = px.line(df, x=df.index, y=['MACD', 'MACD_Signal'], title='📈 MACD y Señal')
                fig_macd.update_layout(template="plotly_white")
                st.plotly_chart(fig_macd, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Métricas actuales")
            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric(f"💰 Precio actual {symbol}", f"${precio:,.2f}")
            col_met2.metric("📉 RSI actual", f"{rsi:.2f}")
            col_met3.metric("📊 EMA 12 actual", f"${ema_12:,.2f}")
            col_met1.metric("📊 EMA 26 actual", f"${ema_26:,.2f}")
            col_met2.metric("📈 MACD actual", f"{macd:.4f}")
            col_met3.metric("📈 MACD Señal actual", f"{macd_signal_val:.4f}")

            st.markdown("---")
            if modo == "Prueba":
                st.subheader("💼 Portafolio ficticio")

                col_pf1, col_pf2, col_pf3 = st.columns(3)
                col_pf1.metric("💵 USDT disponible", f"${st.session_state['usdt']:,.2f}")
                cripto = symbol.split('-')[0]
                col_pf2.metric(f"🪙 {cripto} en cartera", f"{st.session_state['btc']:.6f}")
                valor_total = st.session_state['usdt'] + st.session_state['btc'] * precio
                col_pf3.metric("📊 Valor total del portafolio", f"${valor_total:,.2f}")

                st.write(f"### Operar {symbol}")
                col_buy, col_sell = st.columns(2)

                with col_buy:
                    usdt_para_comprar = st.number_input(
                        f"Cantidad USDT a usar para comprar {cripto}",
                        min_value=0.0, max_value=st.session_state['usdt'], step=10.0, format="%.2f"
                    )
                    if st.button(f"Comprar {cripto}"):
                        costo_con_comision = usdt_para_comprar * (1 + COMISION)
                        if 0 < usdt_para_comprar <= st.session_state['usdt'] and costo_con_comision <= st.session_state['usdt']:
                            btc_comprado = (usdt_para_comprar / precio) * (1 - COMISION)
                            st.session_state['usdt'] -= costo_con_comision
                            st.session_state['btc'] += btc_comprado
                            st.session_state['historial'].append({
                                'tipo': 'Compra',
                                'hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'precio': precio,
                                'btc': btc_comprado,
                                'usdt': -costo_con_comision
                            })
                            st.success(f"Compra simulada: {btc_comprado:.6f} {cripto} comprado por ${costo_con_comision:.2f}")
                        else:
                            st.warning("No tienes suficiente USDT para cubrir la compra y la comisión.")

                with col_sell:
                    btc_para_vender = st.number_input(
                        f"Cantidad {cripto} a vender",
                        min_value=0.0, max_value=st.session_state['btc'], step=0.0001, format="%.6f"
                    )
                    if st.button(f"Vender {cripto}"):
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
                            st.success(f"Venta simulada: {cripto} vendido {btc_para_vender:.6f}, USDT recibido neto {ingreso_usdt:.2f}")
                        else:
                            st.warning("Ingresa una cantidad válida para vender.")

                if st.session_state['historial']:
                    with st.expander("📝 Historial de operaciones"):
                        df_hist = pd.DataFrame(st.session_state['historial'])
                        st.dataframe(df_hist)

            else:
                st.info("Modo Real no implementado en este demo.")

        except Exception as e:
            st.error(f"Error al procesar datos: {e}")

    else:
        st.warning("No se pudieron obtener datos de BTC.")


with tabs[1]:
    st.subheader("📈 Backtesting de señales combinadas")

    # Comisión de la operación (ajustar según tu caso)
    COMISION = 0.001
    capital_inicial = 1000  # Definir capital inicial (usado más adelante)

    def generar_señales_historicas(df, rsi_venta=70, rsi_compra=30, comision=0.001):
        posicion_actual = "SIN_SEÑAL"
        señales = []
        for i in range(1, len(df)):
            rsi = df['RSI'].iloc[i]
            ema12 = df['EMA_rapida'].iloc[i]
            ema26 = df['EMA_lenta'].iloc[i]
            ema12_ayer = df['EMA_rapida'].iloc[i - 1]
            ema26_ayer = df['EMA_lenta'].iloc[i - 1]
            macd = df['MACD'].iloc[i]
            macd_signal = df['MACD_Signal'].iloc[i]
            macd_ayer = df['MACD'].iloc[i - 1]
            macd_signal_ayer = df['MACD_Signal'].iloc[i - 1]

            # Saltar si hay valores NaN en indicadores
            if np.isnan([rsi, ema12, ema26, macd, macd_signal]).any():
                señales.append("SIN_SEÑAL")
                continue

            señal_rsi = 1 if rsi < rsi_compra else -1 if rsi > rsi_venta else 0
            señal_ema = 1 if (ema12_ayer < ema26_ayer and ema12 > ema26) else -1 if (ema12_ayer > ema26_ayer and ema12 < ema26) else 0
            señal_macd = 1 if (macd_ayer < macd_signal_ayer and macd > macd_signal) else -1 if (macd_ayer > macd_signal_ayer and macd < macd_signal) else 0

            suma = señal_rsi + señal_ema + señal_macd

            if suma > 0 and posicion_actual != "COMPRA":
                señal = "COMPRA"
                posicion_actual = "COMPRA"
            elif suma < 0 and posicion_actual != "VENTA":
                señal = "VENTA"
                posicion_actual = "VENTA"
            else:
                señal = "SIN_SEÑAL"

            señales.append(señal)

        return ["SIN_SEÑAL"] + señales  # La primera fila no tiene señal

    def evaluar_estrategia(df, capital_inicial=1000, comision=0.001):
        capital = capital_inicial
        btc = 0
        for i in range(len(df)):
            señal = df["Señal"].iloc[i]
            precio = df["Close"].iloc[i]
            if señal == "COMPRA" and capital > 0:
                btc = (capital / precio) * (1 - comision)
                capital = 0
            elif señal == "VENTA" and btc > 0:
                capital = (btc * precio) * (1 - comision)
                btc = 0
        # Valor total (capital en cash + valor de BTC al último precio)
        return capital + btc * df["Close"].iloc[-1]

    # Valores iniciales para indicadores (pueden venir de parámetros externos)
    rsi_periodo = 14
    ema_rapida = 12
    ema_lenta = 26
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_compra = 30
    rsi_venta = 70

    # Rango para optimización
    rsi_rango = [8, 14, 20]
    ema_rapida_rango = [6, 12, 15]
    ema_lenta_rango = [18, 26, 30]
    macd_fast_rango = [8, 10, 12]
    macd_slow_rango = [22, 26, 28]
    macd_signal_rango = [7, 11]
    rsi_compra_rango = range(20, 31, 5)  
    rsi_venta_rango = range(70, 81, 5)  

    # Cargar datos
    df = obtener_datos(symbol, period, rsi_periodo, ema_rapida, ema_lenta, macd_fast, macd_slow, macd_signal)
    # Si tiene MultiIndex en columnas, seleccionar primer nivel
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    split_index = int(len(df) * 0.7)
    df_train = df.iloc[:split_index].copy()
    df_test = df.iloc[split_index:].copy()

    mejor_resultado = -float('inf')
    mejores_parametros = None

    # Optimización en datos de entrenamiento
    for rsi_p in rsi_rango:
        for rsi_compra in rsi_compra_rango:
            for rsi_venta in rsi_venta_rango:
                for ema_f in ema_rapida_rango:
                    for ema_l in ema_lenta_rango:
                        if ema_f >= ema_l:
                            continue
                        for macd_f in macd_fast_rango:
                            for macd_s in macd_slow_rango:
                                if macd_f >= macd_s:
                                    continue
                                for macd_sig in macd_signal_rango:
                                    # Calcular indicadores
                                    df_train['RSI'] = ta.momentum.rsi(df_train['Close'], window=rsi_p)
                                    df_train['EMA_rapida'] = ta.trend.ema_indicator(df_train['Close'], window=ema_f)
                                    df_train['EMA_lenta'] = ta.trend.ema_indicator(df_train['Close'], window=ema_l)

                                    macd = ta.trend.MACD(df_train['Close'], window_slow=macd_s, window_fast=macd_f, window_sign=macd_sig)
                                    df_train['MACD'] = macd.macd()
                                    df_train['MACD_Signal'] = macd.macd_signal()

                                    df_train.dropna(inplace=True)
                                    # Pasar umbrales a la función de generación de señales
                                    df_train["Señal"] = generar_señales_historicas(df_train, rsi_compra=rsi_compra, rsi_venta=rsi_venta)
                                    capital_final = evaluar_estrategia(df_train, capital_inicial, COMISION)

                                    if capital_final > mejor_resultado:
                                        mejor_resultado = capital_final
                                        mejores_parametros = (rsi_p, rsi_compra, rsi_venta, ema_f, ema_l, macd_f, macd_s, macd_sig)


    if mejores_parametros is None:
        st.error("No se encontraron parámetros óptimos con los rangos especificados.")
    else:
        # Desempaquetar parámetros
        rsi_p, ema_f, ema_l, macd_f, macd_s, macd_sig = mejores_parametros

        # Calcular indicadores en test con mejores parámetros
        df_test['RSI'] = ta.momentum.rsi(df_test['Close'], window=rsi_p)
        df_test['EMA_rapida'] = ta.trend.ema_indicator(df_test['Close'], window=ema_f)
        df_test['EMA_lenta'] = ta.trend.ema_indicator(df_test['Close'], window=ema_l)

        macd = ta.trend.MACD(df_test['Close'], window_slow=macd_s, window_fast=macd_f, window_sign=macd_sig)
        df_test['MACD'] = macd.macd()
        df_test['MACD_Signal'] = macd.macd_signal()

        df_test.dropna(inplace=True)
        df_test["Señal"] = generar_señales_historicas(df_test)

        # Tomar últimas 100 filas para mostrar
        df_test_formateado = df_test[["Close", "RSI", "EMA_rapida", "EMA_lenta", "MACD", "MACD_Signal", "Señal"]].tail(100).copy()

        capital_final_test = evaluar_estrategia(df_test_formateado, capital_inicial, COMISION)

        # Mostrar tabla formateada (solo visual)
        df_test_formateado_mostrar = df_test_formateado.copy()
        for col in ["Close", "EMA_rapida", "EMA_lenta"]:
            df_test_formateado_mostrar[col] = df_test_formateado_mostrar[col].apply(lambda x: f"${x:,.2f}")

        st.write("#### Señales generadas:")
        st.dataframe(df_test_formateado_mostrar)

        st.info(f"📊 Mejores parámetros encontrados: RSI={rsi_p}, EMA rápida={ema_f}, EMA lenta={ema_l}, MACD fast={macd_f}, slow={macd_s}, signal={macd_sig}")
        st.success(f"💼 Capital final tras simulación en Test: ${capital_final_test:,.2f}")

        print(f"Mejores parámetros: RSI={rsi_p}, EMA rápida={ema_f}, EMA lenta={ema_l}, MACD fast={macd_f}, slow={macd_s}, signal={macd_sig}")
        print(f"Capital final en test: ${capital_final_test:,.2f}")

        # Gráfico con plotly
        fig = go.Figure()

        # Línea Close
        fig.add_trace(go.Scatter(
            x=df_test.index,
            y=df_test['Close'],
            mode='lines',
            name='Precio Close',
            line=dict(color='blue')
        ))

        # Puntos Compra
        fig.add_trace(go.Scatter(
            x=df_test[df_test['Señal'] == 'COMPRA'].index,
            y=df_test[df_test['Señal'] == 'COMPRA']['Close'],
            mode='markers',
            marker_symbol='triangle-up',
            marker_color='green',
            marker_size=12,
            name='Compra'
        ))

        # Puntos Venta
        fig.add_trace(go.Scatter(
            x=df_test[df_test['Señal'] == 'VENTA'].index,
            y=df_test[df_test['Señal'] == 'VENTA']['Close'],
            mode='markers',
            marker_symbol='triangle-down',
            marker_color='red',
            marker_size=12,
            name='Venta'
        ))

        fig.update_layout(
            title=f"Precio Close con Señales de Compra y Venta para {symbol}",
            xaxis_title="Fecha",
            yaxis_title="Precio",
            legend_title="Leyenda",
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Simulación para mostrar operaciones individuales
        capital = capital_inicial
        btc = 0
        historial_test = []

        for i in range(len(df_test)):
            señal = df_test["Señal"].iloc[i]
            precio = df_test["Close"].iloc[i]
            if señal == "COMPRA" and capital > 0:
                btc = (capital / precio) * (1 - COMISION)
                historial_test.append(("COMPRA", df_test.index[i], precio, capital, btc))
                capital = 0
            elif señal == "VENTA" and btc > 0:
                capital = (btc * precio) * (1 - COMISION)
                historial_test.append(("VENTA", df_test.index[i], precio, capital, btc))
                btc = 0

        if historial_test:
            df_historial = pd.DataFrame(historial_test, columns=["Tipo", "Fecha", "Precio", "Capital", "BTC"])
            st.write("#### Historial de operaciones en test")
            st.dataframe(df_historial)
        else:
            st.info("No se generaron operaciones en el período de Test con la estrategia actual.")


