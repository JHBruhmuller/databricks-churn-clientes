"""Gera Churn_Databricks_Colab.ipynb a partir de notebooks/churn_databricks.py.

O código do pipeline é o mesmo; só mudam a preparação do ambiente (PySpark + Java no Colab)
e a origem do CSV (baixado do GitHub).
"""
import json
from pathlib import Path

RAIZ = Path(__file__).parent
USUARIO_REPO = "JHBruhmuller/databricks-churn-clientes"
CSV_URL = f"https://raw.githubusercontent.com/{USUARIO_REPO}/main/dados/churn_clientes.csv"

fonte = (RAIZ / "notebooks" / "churn_databricks.py").read_text(encoding="utf-8")
blocos = [b.strip("\n") for b in fonte.split("# COMMAND ----------")]
blocos[0] = blocos[0].replace("# Databricks notebook source", "").strip("\n")


def celula(tipo, texto):
    c = {"cell_type": tipo, "metadata": {}, "source": texto.strip("\n").splitlines(keepends=True)}
    if tipo == "code":
        c.update(execution_count=None, outputs=[])
    return c


def para_markdown(bloco):
    linhas = []
    for l in bloco.splitlines():
        l = l.removeprefix("# MAGIC").removeprefix(" ")
        linhas.append(l.removeprefix("%md").lstrip() if l.startswith("%md") else l)
    return "\n".join(linhas)


intro = f"""# Previsão de churn de clientes (telecom) com Spark MLlib

Versão para **Google Colab** do notebook do Databricks (`notebooks/churn_databricks.py`): mesmo pipeline de ingestão, ETL, Random Forest e interpretação. Execute com **Ambiente de execução > Executar tudo**.

No Colab o Spark roda em modo local e as tabelas são gravadas em Parquet; no Databricks elas são Delta Lake."""

setup = f'''# Preparação do ambiente: Java 17 e PySpark (leva cerca de 1 minuto)
!apt-get -qq install -y openjdk-17-jdk-headless > /dev/null
!pip -q install pyspark "pandas<3"

import os, urllib.request
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"

from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("churn").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

from IPython.display import display as _display
def display(obj):  # equivalente ao display() do Databricks
    _display(obj.limit(10).toPandas() if hasattr(obj, "toPandas") else obj)

urllib.request.urlretrieve("{CSV_URL}", "churn_clientes.csv")'''

celulas = [celula("markdown", intro), celula("code", setup)]
for b in blocos:
    if b.startswith("# MAGIC"):
        if "Previsão de churn" in b:  # título original, substituído pela introdução do Colab
            continue
        celulas.append(celula("markdown", para_markdown(b)))
    else:
        celulas.append(celula("code", b.replace('CAMINHO_CSV = "/Volumes/workspace/default/churn/churn_clientes.csv"',
                                                'CAMINHO_CSV = "churn_clientes.csv"')))

nb = {"cells": celulas, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(RAIZ / "Churn_Databricks_Colab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(celulas)} células geradas")
