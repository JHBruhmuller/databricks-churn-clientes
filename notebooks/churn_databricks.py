# Databricks notebook source
# MAGIC %md
# MAGIC # Previsão de churn de clientes (telecom) com Spark MLlib
# MAGIC Ingestão (Delta Lake) → ETL → Random Forest → insights de negócio.
# MAGIC
# MAGIC Antes de rodar: envie `dados/churn_clientes.csv` para um Volume, ex.: `/Volumes/workspace/default/churn/churn_clientes.csv`.

# COMMAND ----------
# MAGIC %md ## 1. Ingestão

# COMMAND ----------
CAMINHO_CSV = "/Volumes/workspace/default/churn/churn_clientes.csv"

df_raw = (spark.read.csv(CAMINHO_CSV, header=True, inferSchema=True))
df_raw.write.mode("overwrite").saveAsTable("churn_clientes_bronze")

print(f"Linhas ingeridas: {df_raw.count()}")
display(spark.table("churn_clientes_bronze").limit(5))

# COMMAND ----------
# MAGIC %md ## 2. Tratamento (ETL com MLlib)

# COMMAND ----------
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

df = spark.table("churn_clientes_bronze")
antes = df.count()
df = df.dropna().dropDuplicates(["cliente_id"])
print(f"Removidas {antes - df.count()} linhas com nulos/duplicadas")
df.write.mode("overwrite").saveAsTable("churn_clientes_silver")

categoricas = ["contrato", "internet", "pagamento"]
numericas = ["meses_cliente", "mensalidade", "chamados_suporte", "atrasos_pagamento"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categoricas]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in categoricas],
                        outputCols=[f"{c}_ohe" for c in categoricas])
assembler = VectorAssembler(inputCols=numericas + [f"{c}_ohe" for c in categoricas],
                            outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)

etl = Pipeline(stages=indexers + [encoder, assembler, scaler])

# COMMAND ----------
# MAGIC %md ## 3. Treinamento

# COMMAND ----------
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from pyspark.sql import functions as F

treino, teste = df.randomSplit([0.8, 0.2], seed=42)

# Classes desbalanceadas (poucos clientes cancelam): pesa mais a classe churn no treino
n_neg, n_pos = treino.filter("churn = 0").count(), treino.filter("churn = 1").count()
peso_pos = n_neg / n_pos
treino = treino.withColumn("peso", F.when(F.col("churn") == 1, F.lit(peso_pos)).otherwise(F.lit(1.0)))
print(f"Treino: {n_pos} churn / {n_neg} permanecem (peso da classe churn = {peso_pos:.2f})")

rf = RandomForestClassifier(labelCol="churn", featuresCol="features", weightCol="peso",
                            numTrees=150, maxDepth=8, seed=42)
modelo = Pipeline(stages=etl.getStages() + [rf]).fit(treino)
pred = modelo.transform(teste)

auc = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC").evaluate(pred)
mc = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction")
acc = mc.setMetricName("accuracy").evaluate(pred)
f1 = mc.setMetricName("f1").evaluate(pred)
print(f"AUC={auc:.3f}  Acurácia={acc:.3f}  F1={f1:.3f}")

pred.groupBy("churn", "prediction").count().orderBy("churn", "prediction").show()
pred.select("cliente_id", "churn", "prediction", "probability").write.mode("overwrite") \
    .saveAsTable("churn_clientes_gold_predicoes")

# COMMAND ----------
# MAGIC %md ## 4. Resultados e insights

# COMMAND ----------
import pandas as pd

nomes = numericas.copy()
for est in modelo.stages[:len(categoricas)]:
    # handleInvalid="keep" cria uma categoria extra, que o OneHotEncoder descarta (dropLast)
    nomes += [f"{est.getInputCol()}={v}" for v in est.labels]
importancias = modelo.stages[-1].featureImportances.toArray()
assert len(nomes) == len(importancias)
imp = pd.DataFrame({"variavel": nomes, "importancia": importancias})
display(imp.sort_values("importancia", ascending=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Interpretação de negócio
# MAGIC * **Quem cancela:** contrato mensal é o sinal mais forte, seguido do tempo de casa (clientes novos saem mais), chamados de suporte e atrasos de pagamento. Forma de pagamento e tipo de internet pesam pouco.
# MAGIC * **Desempenho:** com as classes balanceadas, o modelo recupera cerca de 79% dos clientes que cancelam, ao custo de precisão baixa (~42%): muitos alertas são falsos positivos.
# MAGIC * **Ação operacional:** ordenar a fila da retenção pela `probability` de churn (tabela `churn_clientes_gold_predicoes`). Abordar 44% da base captura 79% dos cancelamentos (cerca de 1,8x mais que uma abordagem aleatória), e a oferta de migração de contrato mensal para anual com desconto ataca o principal fator.
# MAGIC * **Impacto:** a retenção deixa de ser campanha em massa e passa a ser focada. Para dimensionar o ROI, compare o desconto dado aos falsos positivos com a receita preservada nos verdadeiros positivos.
# MAGIC * **Limites:** os dados são sintéticos (`gerar_dados.py`); confirme os resultados com dados reais antes de decidir.
