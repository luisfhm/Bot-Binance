import pandas as pd

def calcular_rsi(data, window):
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calcular_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calcular_ema(data, fast)
    ema_slow = calcular_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calcular_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
