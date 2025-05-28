from estrategia import generar_señales_historicas

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
    return capital + btc * df["Close"].iloc[-1]
