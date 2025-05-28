import numpy as np

def calcular_senal_actual(df):
    rsi = df['RSI'].iloc[-1]
    ema_rapida = df['EMA_rapida'].iloc[-1]
    ema_lenta = df['EMA_lenta'].iloc[-1]
    ema_rapida_ayer = df['EMA_rapida'].iloc[-2]
    ema_lenta_ayer = df['EMA_lenta'].iloc[-2]
    macd = df['MACD'].iloc[-1]
    macd_ayer = df['MACD'].iloc[-2]
    macd_signal = df['MACD_Signal'].iloc[-1]
    macd_signal_ayer = df['MACD_Signal'].iloc[-2]

    senal_rsi = 1 if rsi < 30 else -1 if rsi > 70 else 0
    senal_ema = 1 if (ema_rapida_ayer < ema_lenta_ayer and ema_rapida > ema_lenta) else -1 if (ema_rapida_ayer > ema_lenta_ayer and ema_rapida < ema_lenta) else 0
    senal_macd = 1 if (macd_ayer < macd_signal_ayer and macd > macd_signal) else -1 if (macd_ayer > macd_signal_ayer and macd < macd_signal) else 0

    suma = senal_rsi + senal_ema + senal_macd

    if suma > 0:
        return "🟢 Señal combinada de COMPRA"
    elif suma < 0:
        return "🔴 Señal combinada de VENTA"
    else:
        return "⚪ Sin señal clara"


def generar_señales_historicas(df, rsi_venta=80, rsi_compra=20):
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

    return ["SIN_SEÑAL"] + señales
