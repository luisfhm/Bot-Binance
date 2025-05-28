import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from indicadores import calcular_rsi, calcular_ema, calcular_macd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ta

def obtener_datos_con_indicadores(symbol="BTC-USD", interval="5m", period="1d"):
    df = yf.download(symbol, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['RSI'] = calcular_rsi(df['Close'], 14)
    df['EMA_rapida'] = calcular_ema(df['Close'], 12)
    df['EMA_lenta'] = calcular_ema(df['Close'], 26)
    macd_line, macd_signal, _ = calcular_macd(df['Close'], 12, 26, 9)
    df['MACD'] = macd_line
    df['MACD_Signal'] = macd_signal
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df.dropna(inplace=True)
    return df


def preparar_dataset(df, horizonte=6):
    df['target'] = (df['Close'].shift(-horizonte) > df['Close']).astype(int)
    features = ['RSI', 'EMA_rapida', 'EMA_lenta', 'MACD', 'MACD_Signal',"SMA_20", "EMA_50", "ATR"]
    df_model = df[features + ['target']].dropna()
    X = df_model[features]
    y = df_model['target']
    return X, y


def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))
    return model


def predecir_ultima(df, model):
    ultimos = df[['RSI', 'EMA_rapida', 'EMA_lenta', 'MACD', 'MACD_Signal',"SMA_20","EMA_50","ATR"]].iloc[-1:]
    pred = model.predict(ultimos)[0]
    return "COMPRA" if pred == 1 else "VENTA"


def evaluar_modelo_en_test(df, model, horizonte=6):
    features = ['RSI', 'EMA_rapida', 'EMA_lenta', 'MACD', 'MACD_Signal',"SMA_20","EMA_50","ATR"]
    df_test = df.copy()
    df_test['target'] = (df_test['Close'].shift(-horizonte) > df_test['Close']).astype(int)
    df_test.dropna(inplace=True)

    df_test['prediccion'] = model.predict(df_test[features])

    capital = 1000
    btc = 0
    historial = []
    capital_hist = []

    for i in range(len(df_test)):
        pred = df_test.iloc[i]['prediccion']
        precio = df_test.iloc[i]['Close']
        if pred == 1 and capital > 0:
            btc = (capital / precio) * (1 - 0.001)
            capital = 0
            historial.append(('COMPRA', df_test.index[i], precio))
        elif pred == 0 and btc > 0:
            capital = (btc * precio) * (1 - 0.001)
            btc = 0
            historial.append(('VENTA', df_test.index[i], precio))
        capital_total = capital + btc * precio
        capital_hist.append(capital_total)

    capital_final = capital + btc * df_test['Close'].iloc[-1]
    print(f"\n💼 Capital final simulado en test: ${capital_final:.2f}")

    # Devolver todo el DataFrame con indicadores para graficar
    return df_test, historial, capital_hist

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
import pandas as pd
import webbrowser

def graficar_resultados_interactivos(df_test, historial, capital_hist, y_test, y_pred):
    # 1. Gráfico interactivo
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Precio Close y operaciones", "RSI", "MACD", "Capital Simulado")
    )

    # Precio Close con señales
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Close'], mode='lines', name='Close'), row=1, col=1)

    compras = [op for op in historial if op[0] == 'COMPRA']
    ventas = [op for op in historial if op[0] == 'VENTA']

    if compras:
        fig.add_trace(
            go.Scatter(
                x=[c[1] for c in compras],
                y=[c[2] for c in compras],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='Compra'
            ),
            row=1, col=1
        )

    if ventas:
        fig.add_trace(
            go.Scatter(
                x=[v[1] for v in ventas],
                y=[v[2] for v in ventas],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='Venta'
            ),
            row=1, col=1
        )

    # RSI
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD y Señal
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['MACD'], mode='lines', name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_test.index, y=df_test['MACD_Signal'], mode='lines', name='MACD Signal'), row=3, col=1)

    # Capital simulado
    fig.add_trace(go.Scatter(x=df_test.index, y=capital_hist, mode='lines', name='Capital Simulado'), row=4, col=1)

    fig.update_layout(height=900, width=1000, title_text="Análisis y Simulación con indicadores técnicos")

    # 2. Tabla de evaluación del modelo
    reporte = classification_report(y_test, y_pred, output_dict=True)
    df_reporte = pd.DataFrame(reporte).transpose().round(2)
    tabla_html = df_reporte.to_html(classes="table table-bordered", border=0)

    # 3. Exportar todo como HTML
    html_output = f"""
    <html>
        <head>
            <title>Resultados Interactivos</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .table {{
                    border-collapse: collapse;
                    width: 60%;
                    margin-top: 30px;
                }}
                .table th, .table td {{
                    border: 1px solid #999;
                    padding: 8px;
                    text-align: center;
                }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Resultados de la Estrategia</h1>
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
            <h2>Evaluación del Modelo en la Muestra Test</h2>
            {tabla_html}
        </body>
    </html>
    """

    with open("resultados_interactivos.html", "w", encoding="utf-8") as f:
        f.write(html_output)

    webbrowser.open("resultados_interactivos.html")

from sklearn.model_selection import train_test_split

df = obtener_datos_con_indicadores()
X, y = preparar_dataset(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

df_test = df.iloc[-len(y_test):].copy()
df_test['target'] = y_test
df_test['prediccion'] = y_pred

df_test, historial, capital_hist = evaluar_modelo_en_test(df, model)
graficar_resultados_interactivos(df_test, historial, capital_hist, y_test, y_pred)



