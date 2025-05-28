import plotly.express as px
import plotly.graph_objects as go

def graficar_precio(df, symbol):
    fig = px.line(df, x=df.index, y='Close', title=f'Precio {symbol}')
    fig.update_layout(template="plotly_white")
    return fig

def graficar_ema(df):
    fig = px.line(df, x=df.index, y=['EMA_rapida', 'EMA_lenta'], title='EMAs')
    fig.update_layout(template="plotly_white")
    return fig

def graficar_rsi(df):
    fig = px.line(df, x=df.index, y='RSI', title='RSI')
    fig.update_layout(template="plotly_white")
    return fig

def graficar_macd(df):
    fig = px.line(df, x=df.index, y=['MACD', 'MACD_Signal'], title='MACD y Señal')
    fig.update_layout(template="plotly_white")
    return fig



def graficar_backtest_entradas_salidas(df, symbol):
    fig = go.Figure()

    # Línea de precio
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Precio',
        line=dict(color='blue')
    ))

    # Señales de COMPRA
    fig.add_trace(go.Scatter(
        x=df[df['Señal'] == 'COMPRA'].index,
        y=df[df['Señal'] == 'COMPRA']['Close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=10),
        name='Compra'
    ))

    # Señales de VENTA
    fig.add_trace(go.Scatter(
        x=df[df['Señal'] == 'VENTA'].index,
        y=df[df['Señal'] == 'VENTA']['Close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=10),
        name='Venta'
    ))

    fig.update_layout(
        title=f"Precio y señales de entrada/salida para {symbol}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_white"
    )

    return fig
