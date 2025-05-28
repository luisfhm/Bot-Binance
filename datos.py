import yfinance as yf
from indicadores import calcular_rsi, calcular_ema, calcular_macd

def obtener_datos(symbol, interval, period):
    # Valores internos predefinidos
    rsi_p = 14
    ema_f = 12
    ema_l = 26
    macd_f = 12
    macd_s = 26
    macd_sig = 9

    df = yf.download(tickers=symbol, interval=interval, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['RSI'] = calcular_rsi(df['Close'], rsi_p)
    df['EMA_rapida'] = calcular_ema(df['Close'], ema_f)
    df['EMA_lenta'] = calcular_ema(df['Close'], ema_l)
    macd_line, signal_line, hist = calcular_macd(df['Close'], macd_f, macd_s, macd_sig)
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Hist'] = hist
    return df

