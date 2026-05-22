"""Build M2_Spark_ML.ipynb from cell sources.

Run from the repo root:
    python scripts/build_notebook.py
"""
import json
import pathlib

NB_FILE = "M2_Spark_ML.ipynb"

REEMA  = "Reema Aldwehi"
LUJAIN = "Lujain Malharbi"


_blocks = []


def md(s):
    _blocks.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": s.splitlines(keepends=True),
    })


def code(s):
    _blocks.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.splitlines(keepends=True),
    })


# ============================================================
md(f"""# SE446 Milestone 2 — Software Group

Spark DataFrame analytics + MLlib arrest predictor for Chicago Crime data.

| Member          | Student ID | GitHub        | Tasks       |
|-----------------|-----------:|---------------|-------------|
| {REEMA}     |     231207 | `Sara-Alsh3`  | 1, 3, 5, 6, 11 |
| {LUJAIN}    |     231300 | `saramst`     | 2, 4, 7, 9, 10 |

**Spec compliance — May 2026 update:**
* Task 8 (CrossValidator) is **waived** by the instructor and is not in this notebook.
* Phase B (Tasks 5–7) trains on a **5% sample** via `df.sample(fraction=0.05, seed=42)`.
* Task 11 uses `--deploy-mode cluster`. Application stdout is collected with
  `yarn logs -applicationId <appId>` into `output/spark_submit/run.log`.
""")


# ------- Setup -------
md("---\n## Step 0 — Spark session bootstrap")

code("""import os
import time
import shutil

import pyspark.sql.functions as spkfn
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import IntegerType, StringType


_HDFS_BIN_ON_PATH = shutil.which("hdfs") is not None


def _start_spark() -> SparkSession:
    settings = (SparkSession.builder
                .appName("M2_Software")
                .config("spark.sql.shuffle.partitions", "8"))
    if _HDFS_BIN_ON_PATH:
        return settings.getOrCreate()
    return (settings
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate())


platform = "cluster" if _HDFS_BIN_ON_PATH else "local"
spark    = _start_spark()
if platform == "local":
    spark.sparkContext.setLogLevel("WARN")

print("Platform:       ", platform)
print("Spark version:  ", spark.version)
print("Spark master:   ", spark.sparkContext.master)
""")


# ------- Data ingestion -------
md("---\n## Step 1 — Read the dataset")

code("""HDFS_FILE = "hdfs:///data/chicago_crimes.csv"


def _ingest_real():
    raw = spark.read.csv(HDFS_FILE, header=True, inferSchema=True)
    return (raw
            .withColumn("Hour",
                        spkfn.hour(spkfn.to_timestamp(spkfn.col("Date"),
                                                  "MM/dd/yyyy hh:mm:ss a")))
            .withColumn("label",        spkfn.col("Arrest").cast(IntegerType()))
            .withColumn("Domestic_str", spkfn.col("Domestic").cast(StringType())))


def _ingest_synthetic(rows: int = 10_000):
    import random
    random.seed(42)
    rate_by_kind = {
        "NARCOTICS":           0.85,
        "PROSTITUTION":        0.80,
        "WEAPONS VIOLATION":   0.60,
        "BATTERY":             0.30,
        "ASSAULT":             0.25,
        "ROBBERY":             0.15,
        "THEFT":               0.10,
        "BURGLARY":            0.08,
        "MOTOR VEHICLE THEFT": 0.06,
        "CRIMINAL DAMAGE":     0.05,
    }
    locs = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER",
            "PARKING LOT", "SCHOOL", "ALLEY", "RESIDENCE-GARAGE"]
    yrs = [2020, 2021, 2022, 2023, 2024, 2025]
    items = []
    for _ in range(rows):
        kind = random.choice(list(rate_by_kind))
        h = random.randint(0, 23)
        is_dom = random.random() < 0.15
        p = rate_by_kind[kind] + (0.20 if is_dom else 0.0)
        if 2 <= h <= 5:
            p -= 0.10
        p = max(0.01, min(0.99, p))
        items.append(Row(
            District=random.randint(1, 25),
            **{"Primary Type": kind},
            **{"Location Description": random.choice(locs)},
            Year=random.choice(yrs),
            Hour=h,
            Domestic_str=str(is_dom).lower(),
            Arrest=random.random() < p,
            label=int(random.random() < p),
        ))
    return spark.createDataFrame(items)


dataset = _ingest_real() if platform == "cluster" else _ingest_synthetic()
dataset.cache()
print("Records ingested:", f"{dataset.count():,}")
dataset.printSchema()
dataset.show(3, truncate=False)
""")


# ============================================================
md("---\n# Phase A — Spark DataFrame analytics")


md(f"""## Task 1 — Crime type distribution
*{REEMA}*

DataFrame `groupBy` + descending count.""")

code(f"""# Task 1 — {REEMA}
crime_freq = (dataset
                   .groupBy("Primary Type")
                   .agg(spkfn.count(spkfn.lit(1)).alias("tally"))
                   .orderBy(spkfn.col("tally").desc()))
crime_freq.show(10, truncate=False)
""")


md(f"""## Task 2 — Location hotspots (Spark SQL)
*{LUJAIN}*

Use `createOrReplaceTempView` and run the query through `spark.sql`.""")

code(f"""# Task 2 — {LUJAIN}
dataset.createOrReplaceTempView("cc_view")

top_spots = spark.sql(\"\"\"
    SELECT  `Location Description` AS where_at,
            COUNT(*)               AS reports
      FROM  cc_view
     WHERE  `Location Description` IS NOT NULL
     GROUP  BY `Location Description`
     ORDER  BY reports DESC
     LIMIT  10
\"\"\")
top_spots.show(truncate=False)
""")


md(f"""## Task 3 — Year trend
*{REEMA}*

Counts per year, with a matplotlib chart on local mode.""")

code(f"""# Task 3 — {REEMA}
yearly = (dataset
                   .groupBy("Year")
                   .agg(spkfn.count(spkfn.lit(1)).alias("incidents"))
                   .orderBy("Year"))
yearly.show(30)
""")

code(f"""# Task 3 chart — {REEMA}
if platform == "local":
    import matplotlib.pyplot as plt

    pdf = yearly.toPandas().dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(pdf["Year"].astype(int), pdf["incidents"], where="mid",
            color="#7d4f50", lw=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents")
    ax.set_title("Chicago crime incidents per year")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/crime_per_year.png", dpi=120)
    plt.show()
else:
    print("Cluster mode — printed table is the deliverable.")
""")


md(f"""## Task 4 — Arrest rate analysis
*{LUJAIN}*

Overall arrest rate plus a per-crime-type breakdown.""")

code(f"""# Task 4 — {LUJAIN}
n_all       = dataset.count()
n_arr = dataset.filter(spkfn.col("Arrest") == True).count()
print(f"Overall arrest rate: {{n_arr:,}} / {{n_all:,}} = {{n_arr/n_all*100:.2f}}%")

by_kind = (dataset
            .groupBy("Primary Type")
            .agg(spkfn.count(spkfn.lit(1)).alias("reports"),
                 spkfn.avg(spkfn.col("label").cast("double")).alias("arrest_rate"))
            .filter(spkfn.col("reports") >= 100)
            .orderBy(spkfn.col("arrest_rate").desc()))
print("Top arrest-rate crime types (min 100 reports):")
by_kind.show(15, truncate=False)
""")


# ============================================================
md("""---
# Phase B — MLlib arrest predictor (5% sample)

Per the May 2026 spec update, Phase B runs on a 5% sample. On the full HDFS
dataset that reduces 793,072 rows to roughly 39,654 — small enough to fit the
cluster's RAM budget. The local 10K synthetic dataset shrinks to ~500 rows.""")

code("""# 5% sample, seed=42, applied before any feature engineering
ml_slice = dataset.sample(fraction=0.05, seed=42)
print("Phase B working set:", f"{ml_slice.count():,} rows  (5% sample, seed=42)")
""")


md(f"""## Task 5 — Feature pipeline
*{REEMA}*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over
four features, 80/20 split with `seed=42`.""")

code(f"""# Task 5 — {REEMA}
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

if "Domestic_str" not in ml_slice.columns:
    ml_slice = ml_slice.withColumn("Domestic_str",
                                     spkfn.col("Domestic").cast(StringType()))

crime_code_idx = StringIndexer(inputCol="Primary Type",
                                  outputCol="crime_code",
                                  handleInvalid="skip")
dom_code_idx   = StringIndexer(inputCol="Domestic_str",
                                  outputCol="dom_code",
                                  handleInvalid="skip")
feat_block        = VectorAssembler(
    inputCols=["crime_code", "Hour", "dom_code", "District"],
    outputCol="feat_block",
)

train_part, test_part = ml_slice.randomSplit([0.8, 0.2], seed=42)
train_part.cache()
test_part.cache()
print("Train rows:", f"{{train_part.count():,}}", " | Test rows:", f"{{test_part.count():,}}")

# Inspect the assembled feature column for 5 rows
inspector = Pipeline(stages=[crime_code_idx, dom_code_idx, feat_block]).fit(train_part)
inspector.transform(train_part).select(
    "Primary Type", "crime_code",
    "Hour",
    "Domestic_str", "dom_code",
    "District",
    "feat_block", "label",
).show(5, truncate=False)
print("Vector layout: [crime_code, Hour, dom_code, District]")
""")


md(f"""## Task 6 — Train and evaluate three classifiers
*{REEMA}*

Logistic Regression (maxIter=100, regParam=0.01), Random Forest (numTrees=100,
maxDepth=5, maxBins=64), GBT (maxIter=50, maxDepth=5, maxBins=64). `maxBins=64`
because Primary Type has more than 32 categories on the cluster.""")

code(f"""# Task 6 helpers — {REEMA}
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

bin_eval   = BinaryClassificationEvaluator(labelCol="label")
multi_eval = MulticlassClassificationEvaluator(labelCol="label",
                                               predictionCol="prediction")


def _measure(predictions):
    return {{
        "AUC":       bin_eval.evaluate(predictions),
        "Accuracy":  multi_eval.evaluate(predictions, {{multi_eval.metricName: "accuracy"}}),
        "F1":        multi_eval.evaluate(predictions, {{multi_eval.metricName: "f1"}}),
        "Precision": multi_eval.evaluate(predictions, {{multi_eval.metricName: "weightedPrecision"}}),
        "Recall":    multi_eval.evaluate(predictions, {{multi_eval.metricName: "weightedRecall"}}),
    }}


def _confusion(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    grid = {{(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}}
    return (grid.get((0, 0), 0), grid.get((0, 1), 0),
            grid.get((1, 0), 0), grid.get((1, 1), 0))
""")

code(f"""# Task 6 training — {REEMA}
learners = [
    ("LogisticRegression",
     LogisticRegression(featuresCol="feat_block", labelCol="label",
                        maxIter=100, regParam=0.01)),
    ("RandomForest",
     RandomForestClassifier(featuresCol="feat_block", labelCol="label",
                            numTrees=100, maxDepth=5,
                            maxBins=64, seed=42)),
    ("GBT",
     GBTClassifier(featuresCol="feat_block", labelCol="label",
                   maxIter=50, maxDepth=5,
                   maxBins=64, seed=42)),
]

board = []
rf_inner   = None
for tag, learner in learners:
    print(f"\\n>> {{tag}}")
    pipe = Pipeline(stages=[crime_code_idx, dom_code_idx, feat_block, learner])
    started = time.time()
    fitted_pipe = pipe.fit(train_part)
    elapsed = time.time() - started
    preds = fitted_pipe.transform(test_part)
    metrics_d = _measure(preds)
    cm = _confusion(preds)
    board.append((tag, fitted_pipe, metrics_d, cm, elapsed))
    for k, v in metrics_d.items():
        print(f"  {{k:<10}}{{v:.4f}}")
    print(f"  Train(s)  {{elapsed:.1f}}")
    print(f"  CM (TN,FP,FN,TP) = {{cm}}")
    if tag == "RandomForest":
        rf_inner = fitted_pipe.stages[-1]

# Comparison
print("\\n" + "=" * 78)
print(f"{{'metric':<11}}{{'Logistic':>14}}{{'RandomForest':>16}}{{'GBT':>14}}")
print("-" * 78)
m_lr, m_rf, m_gbt = (board[0][2], board[1][2], board[2][2])
for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
    print(f"{{k:<11}}{{m_lr[k]:>14.4f}}{{m_rf[k]:>16.4f}}{{m_gbt[k]:>14.4f}}")
print(f"{{'Train(s)':<11}}{{board[0][4]:>14.1f}}{{board[1][4]:>16.1f}}{{board[2][4]:>14.1f}}")
print("=" * 78)
champ = max(board, key=lambda r: r[2]["AUC"])
print(f"Top model by AUC: {{champ[0]}} ({{champ[2]['AUC']:.4f}})")
""")


md(f"""## Task 7 — Random Forest feature importances
*{LUJAIN}*

Importances tell us which feature drives most of the splits in the trees.""")

code(f"""# Task 7 — {LUJAIN}
schema_layout = ["crime_code", "Hour", "dom_code", "District"]
imp_arr = rf_inner.featureImportances.toArray()

print("Random Forest feature importances:")
for col_name, imp in sorted(zip(schema_layout, imp_arr), key=lambda kv: -kv[1]):
    bar = "+" * int(round(imp * 50))
    print(f"  {{col_name:<10}} {{imp:.4f}}  {{bar}}")
""")


md("""**Reading the importances.** The crime-type label dominates because the per-crime
arrest-rate distribution from Task 4 is itself dominated by crime type — NARCOTICS
is near 99% while THEFT is near 14%. Once a tree splits on the crime label it has
most of its answer.

Logistic Regression underperforms the tree models because it treats `crime_code` as
a numeric feature and fits a single linear coefficient, implying a meaningless
ordering between crime types. Trees split on individual values of the index and
side-step that issue entirely.""")


md("""---
## Cleanup""")

code("""spark.stop()""")


# ------- Write -------
nb = {
    "cells": _blocks,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

pathlib.Path(NB_FILE).write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_FILE} ({len(_blocks)} cells)")
