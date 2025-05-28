def comprar(usdt_actual, precio, usdt_a_usar, comision):
    if usdt_a_usar > usdt_actual:
        return usdt_actual, 0.0, False
    costo = usdt_a_usar * (1 + comision)
    btc = (usdt_a_usar / precio) * (1 - comision)
    return usdt_actual - costo, btc, True

def vender(btc_actual, precio, btc_a_vender, comision):
    if btc_a_vender > btc_actual:
        return btc_actual, 0.0, False
    usdt = (btc_a_vender * precio) * (1 - comision)
    return btc_actual - btc_a_vender, usdt, True
