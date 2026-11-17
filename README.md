# Previsão de churn com Databricks, Delta Lake e Spark MLlib

Atividade de Ciência de Dados: pipeline completo numa operadora de telecom fictícia.

**Notebook:** [notebooks/churn_databricks.py](https://github.com/JHBruhmuller/databricks-churn-clientes/blob/main/notebooks/churn_databricks.py)

| Etapa | O que foi feito |
|---|---|
| Ingestão | `spark.read.csv()` → `saveAsTable` (Delta Lake, camada bronze) |
| Tratamento | `dropna()`, `dropDuplicates`, `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler` |
| Treinamento | `randomSplit([0.8, 0.2])` + `RandomForestClassifier` |
| Resultados | AUC, acurácia, F1, importância das variáveis e leitura de negócio |

## Como executar
1. `python gerar_dados.py` (já há um CSV pronto em `dados/`).
2. No Databricks, envie o CSV para um Volume e ajuste `CAMINHO_CSV` no notebook.
3. Importe `notebooks/churn_databricks.py` (Import → arquivo) e execute as células.

Estudos: https://docs.databricks.com/aws/en/getting-started/
