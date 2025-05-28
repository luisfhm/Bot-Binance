import yfinance as yf
from indicadores import calcular_rsi, calcular_ema, calcular_macd

def obtener_datos(symbol, interval, period, rsi_p, ema_f, ema_l, macd_f, macd_s, macd_sig):
    df = yf.download(tickers=symbol, interval=interval, period=period)
    df.columns = df.columns.get_level_values(0)  # Elimina MultiIndex en columnas
    df['RSI'] = calcular_rsi(df['Close'], rsi_p)
    df['EMA_rapida'] = calcular_ema(df['Close'], ema_f)
    df['EMA_lenta'] = calcular_ema(df['Close'], ema_l)
    macd_line, signal_line, hist = calcular_macd(df['Close'], fast=macd_f, slow=macd_s, signal=macd_sig)
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    return df
