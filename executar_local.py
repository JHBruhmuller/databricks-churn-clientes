"""Executa o notebook do Databricks localmente (PySpark) e exporta os resultados.

No Databricks isto não é necessário: basta importar notebooks/churn_databricks.py.
Localmente as tabelas são gravadas em Parquet (Delta Lake é o formato padrão só no Databricks).
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyspark
from pyspark.sql import SparkSession

raiz = Path(__file__).parent
saida = raiz / "resultados"
saida.mkdir(exist_ok=True)

spark = (SparkSession.builder.master("local[*]").appName("churn-local")
         .config("spark.sql.warehouse.dir", tempfile.mkdtemp())
         .config("spark.sql.shuffle.partitions", "8").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


def display(obj):
    (obj.show(10, truncate=False) if hasattr(obj, "show") else print(obj.head(15)))


codigo = (raiz / "notebooks" / "churn_databricks.py").read_text(encoding="utf-8")
csv_local = (raiz / "dados" / "churn_clientes.csv").as_posix()
codigo = re.sub(r'CAMINHO_CSV = ".*"', lambda _: f'CAMINHO_CSV = "{csv_local}"', codigo)
ns = {"spark": spark, "display": display, "__name__": "__main__"}
exec(compile(codigo, "churn_databricks.py", "exec"), ns)

pred, imp = ns["pred"], ns["imp"].sort_values("importancia", ascending=False)
bronze, silver, teste, treino = (ns["df_raw"].count(), ns["df"].count(), ns["teste"].count(), ns["treino"].count())

linhas = pred.groupBy("churn", "prediction").count().toPandas()
linhas.to_csv(saida / "matriz_confusao.csv", index=False)
imp[["variavel", "importancia"]].to_csv(saida / "importancia_variaveis.csv", index=False)
tp = int(linhas.query("churn==1 and prediction==1")["count"].sum())
fn = int(linhas.query("churn==1 and prediction==0")["count"].sum())
fp = int(linhas.query("churn==0 and prediction==1")["count"].sum())
metricas = {
    "auc": round(ns["auc"], 4), "acuracia": round(ns["acc"], 4), "f1": round(ns["f1"], 4),
    "precisao_churn": round(tp / max(tp + fp, 1), 4), "recall_churn": round(tp / max(tp + fn, 1), 4),
    "linhas_bronze": bronze, "linhas_silver": silver, "treino": treino, "teste": teste,
    "taxa_churn_teste": round(pred.filter("churn=1").count() / teste, 4),
    "pyspark": pyspark.__version__, "python": sys.version.split()[0],
}
(saida / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

ns["fig_resultados"].savefig(saida / "graficos_resultados.png", dpi=130)
print(json.dumps(metricas, indent=2, ensure_ascii=False))
spark.stop()
