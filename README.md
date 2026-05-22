# SE446 Milestone 2 — Software Group

Spark DataFrame analytics + MLlib arrest predictor on Chicago Crime data.

## Team

| Member            | GitHub           | Tasks       |
|-------------------|------------------|-------------|
| Reema Aldwehi     | `raldwehi`       | 1, 5, 11    |
| Lujain Malharbi   | `lujainalfaisal` | 2, 6, 9     |
| Leen Althunayan   | `L-alth`         | 3, 4, 7, 10 |

## Spec compliance (May 2026 update)

- Task 8 (CrossValidator) is **omitted** — waived by the instructor.
- Phase B (Tasks 5–7) trains on a **5% sample**: `df.sample(fraction=0.05, seed=42)`.
- Task 11 uses `--deploy-mode cluster`; stdout collected via `yarn logs -applicationId <appId>` into `output/spark_submit/run.log`.

## Repository layout

```
.
├── M2_Spark_ML.ipynb        Notebook (Tasks 1–7)
├── m2_spark_ml.py           Standalone Phase B script for spark-submit
├── scripts/
│   └── build_notebook.py    Notebook generator
└── README.md
```

## Phase A — DataFrame analytics

### Task 1 — Crime type distribution
*Reema Aldwehi (`raldwehi`)*

DataFrame `groupBy` + descending count on `Primary Type`. Numbers match M1 MapReduce exactly on the full 793K-row HDFS dataset.

### Task 2 — Location hotspots (Spark SQL)
*Lujain Malharbi (`lujainalfaisal`)*

`createOrReplaceTempView` + `spark.sql` for top-10 hotspots.

### Task 3 — Year trend
*Leen Althunayan (`L-alth`)*

Yearly crime counts via DataFrame `groupBy` plus matplotlib chart.

### Task 4 — Arrest rate
*Leen Althunayan (`L-alth`)*

Overall arrest rate plus per-crime-type breakdown.

## Phase B — MLlib arrest predictor (5% sample)

### Task 5 — Feature pipeline
*Reema Aldwehi (`raldwehi`)*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over `[crime_code, Hour, dom_code, District]`, 80/20 split with `seed=42`.

### Task 6 — Train and evaluate three classifiers
*Lujain Malharbi (`lujainalfaisal`)*

Logistic Regression (maxIter=100, regParam=0.01), Random Forest (numTrees=100, maxDepth=5, maxBins=64), GBT (maxIter=50, maxDepth=5, maxBins=64).

### Task 7 — Random Forest feature importances
*Leen Althunayan (`L-alth`)*

`primary_type` index dominates because the per-crime arrest-rate distribution is itself dominated by crime type.

## Phase C — Deployment evidence

### Task 9 — Local execution
*Lujain Malharbi (`lujainalfaisal`)*

Notebook executed with `jupyter nbconvert --execute`. Spark master: `local[*]`.

### Task 10 — Cluster execution: client mode
*Leen Althunayan (`L-alth`)*

```bash
ssh user@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 768m --executor-cores 1 \
    --driver-memory 1g notebook_runner.py
```

### Task 11 — spark-submit (cluster mode)
*Reema Aldwehi (`raldwehi`)*

```bash
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```

## Spec note — executor cores

The M2 spec lists `--executor-cores 2`. The course YARN cluster's maximum container allocation is `<memory:1536, vCores:1>` — requesting 2 vcores returns `InvalidResourceRequestException`. We therefore use `--executor-cores 1`, the same setting M1 used.

## Member contributions

| Member | Tasks | Contribution |
|--------|-------|--------------|
| Reema Aldwehi (`raldwehi`) | 1, 5, 11 | Crime-type DataFrame query; feature pipeline; spark-submit cluster mode |
| Lujain Malharbi (`lujainalfaisal`) | 2, 6, 9 | Spark SQL hotspots; three-classifier comparison; local execution evidence |
| Leen Althunayan (`L-alth`) | 3, 4, 7, 10 | Year trend; arrest rate; RF feature importances; yarn-client evidence |
