"""SE446 Milestone 2 — Software Group

Phase B (Tasks 5-7): standalone Spark MLlib pipeline for the spark-submit deliverable.

Authors:
    Task 5 — Reema Aldwehi
    Task 6 — Reema Aldwehi
    Task 7 — Lujain Malharbi

Per the May 2026 spec update:
    * Task 8 (CrossValidator) is waived.
    * Phase B trains on a 5% sample (df.sample(fraction=0.05, seed=42)).

Submit via:
    spark-submit \\
        --master yarn --deploy-mode cluster \\
        --num-executors 2 --executor-memory 1g --executor-cores 1 \\
        m2_spark_ml.py
"""
import time

from pyspark.sql import SparkSession
import pyspark.sql.functions as spkfn
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)


HDFS_FILE = "hdfs:///data/chicago_crimes.csv"


def boot_spark() -> SparkSession:
    return (SparkSession.builder
            .appName("M2_Software_m2_spark_ml")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def fetch_records(session: SparkSession):
    raw = session.read.csv(HDFS_FILE, header=True, inferSchema=True)
    rows = (raw
            .withColumn("Hour",
                        spkfn.hour(spkfn.to_timestamp(spkfn.col("Date"),
                                                  "MM/dd/yyyy hh:mm:ss a")))
            .withColumn("label",        spkfn.col("Arrest").cast(IntegerType()))
            .withColumn("Domestic_str", spkfn.col("Domestic").cast(StringType())))
    return rows.dropna(subset=["District", "Primary Type",
                               "Hour", "Domestic_str", "label"])


def collect_metrics(predictions, bin_eval, multi_eval):
    return {
        "AUC":       bin_eval.evaluate(predictions),
        "Accuracy":  multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"}),
        "F1":        multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}),
        "Precision": multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}),
        "Recall":    multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}),
    }


def confusion_grid(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    grid = {(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}
    return (grid.get((0, 0), 0), grid.get((0, 1), 0),
            grid.get((1, 0), 0), grid.get((1, 1), 0))


def main():
    spark = boot_spark()
    print("Spark version:", spark.version)
    print("Master:       ", spark.sparkContext.master)

    dataset = fetch_records(spark)
    print("Total records:", f"{dataset.count():,}")

    # ----- Task 5 (Sara): pipeline + 5% sample -----
    sub = dataset.sample(fraction=0.05, seed=42)
    print("Phase B sample:", f"{sub.count():,} rows  (5%, seed=42)")

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

    train_part, test_part = sub.randomSplit([0.8, 0.2], seed=42)
    train_part.cache()
    test_part.cache()
    print("Train rows:", f"{train_part.count():,}", "| Test rows:", f"{test_part.count():,}")

    bin_eval   = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label",
                                                   predictionCol="prediction")

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

    rf_inner = None
    bag = []
    for tag, learner in learners:
        pipe = Pipeline(stages=[crime_code_idx, dom_code_idx, feat_block, learner])
        t0 = time.time()
        fitted = pipe.fit(train_part)
        elapsed = time.time() - t0
        preds = fitted.transform(test_part)
        metrics_d = collect_metrics(preds, bin_eval, multi_eval)
        cm = confusion_grid(preds)
        bag.append((tag, elapsed, metrics_d, cm))
        print(f"\n>> {tag}")
        for k, v in metrics_d.items():
            print(f"  {k:<10}{v:.4f}")
        print(f"  Train(s)  {elapsed:.1f}")
        print(f"  CM (TN,FP,FN,TP) = {cm}")
        if tag == "RandomForest":
            rf_inner = fitted.stages[-1]

    print("\n" + "=" * 78)
    print(f"{'metric':<11}{'Logistic':>14}{'RandomForest':>16}{'GBT':>14}")
    print("-" * 78)
    for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
        print(f"{k:<11}{bag[0][2][k]:>14.4f}{bag[1][2][k]:>16.4f}{bag[2][2][k]:>14.4f}")
    print(f"{'Train(s)':<11}{bag[0][1]:>14.1f}{bag[1][1]:>16.1f}{bag[2][1]:>14.1f}")
    print("=" * 78)
    champ = max(bag, key=lambda r: r[2]["AUC"])
    print("Top model by AUC:", champ[0], f"({champ[2]['AUC']:.4f})")

    # ----- Task 7 (Sarah): RF feature importances -----
    print("\n--- Random Forest feature importances ---")
    schema_layout = ["crime_code", "Hour", "dom_code", "District"]
    for col_name, imp in sorted(zip(schema_layout, rf_inner.featureImportances.toArray()),
                                key=lambda kv: -kv[1]):
        print(f"  {col_name:<10} {imp:.4f}  {'+' * int(round(imp * 50))}")

    spark.stop()


if __name__ == "__main__":
    main()
