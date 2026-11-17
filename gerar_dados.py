"""Gera um dataset sintético de churn de uma operadora de telecom (semente fixa)."""
import csv
import random

random.seed(42)
N = 5000
contratos = ["Mensal", "Anual", "Bienal"]
internet = ["Fibra", "DSL", "Nenhuma"]
pagamentos = ["Boleto", "Cartao", "Pix", "Debito"]

with open("dados/churn_clientes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cliente_id", "meses_cliente", "contrato", "internet", "pagamento",
                "mensalidade", "chamados_suporte", "atrasos_pagamento", "churn"])
    for i in range(1, N + 1):
        contrato = random.choices(contratos, [0.55, 0.25, 0.20])[0]
        net = random.choices(internet, [0.45, 0.40, 0.15])[0]
        meses = random.randint(1, 72)
        mensal = round({"Fibra": 120, "DSL": 80, "Nenhuma": 40}[net] + random.gauss(0, 12), 2)
        chamados = max(0, int(random.gauss(2, 2)))
        atrasos = max(0, int(random.gauss(1, 1.3)))
        p = 0.10
        p += 0.30 if contrato == "Mensal" else -0.05
        p += 0.10 if net == "Fibra" and mensal > 130 else 0
        p += 0.04 * chamados + 0.06 * atrasos - 0.004 * meses
        churn = 1 if random.random() < min(max(p, 0.02), 0.95) else 0
        # ~2% de nulos para exercitar o dropna()
        if random.random() < 0.02:
            mensal = ""
        w.writerow([i, meses, contrato, net, random.choice(pagamentos),
                    mensal, chamados, atrasos, churn])
