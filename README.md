# SE446 Milestone 2 — Software Group

Spark DataFrame analytics + MLlib arrest predictor on the Chicago Crime dataset
(Hadoop 3.4.1 / Spark 3.5.4, 1 master + 2 workers).

## Team

| Member            | GitHub           | Tasks       |
|-------------------|------------------|-------------|
| Reema Aldwehi     | `raldwehi`       | 1, 5, 11    |
| Lujain Malharbi   | `lujainalfaisal` | 2, 6, 9     |
| Leen Althunayan   | `L-alth`         | 3, 4, 7, 10 |

## Spec compliance (May 2026 update)

1. **Task 8 (CrossValidator) is omitted** — waived by the instructor.
2. **Phase B (Tasks 5–7) trains on a 5% sample** via `df.sample(fraction=0.05, seed=42)`.
   On the cluster this gives 39,534 rows (Train 31,728 / Test 7,806).
3. **Task 11 uses `--deploy-mode cluster`**. Application stdout is collected with
   `yarn logs -applicationId <appId>` into `output/spark_submit/run.log`.

## Repository layout

```
.
├── M2_Spark_ML.ipynb               Notebook (Tasks 1–7), executed locally
├── m2_spark_ml.py                  Standalone Phase B script for spark-submit
├── scripts/
│   └── build_notebook.py           Notebook generator
├── output/
│   ├── yearly_trend.png            Task 3 chart
│   ├── cluster_yarn_log.txt        Task 10 evidence
│   └── spark_submit/
│       ├── console.log             Task 11 spark-submit console output
│       └── run.log                 Task 11 application stdout
└── README.md
```

## Executive summary

We reproduce the four M1 MapReduce analyses with Spark DataFrames + Spark SQL on the
full 793,072-row HDFS dataset (numbers match M1 exactly). For arrest prediction we
build a Spark MLlib pipeline (StringIndexer × 2 + VectorAssembler + classifier) and
train Logistic Regression, Random Forest, and Gradient-Boosted Trees on a 5% sample
as required by the May 2026 spec update.

**Top model by AUC: GBT (0.8241).** Random Forest is a strong second (0.8057) at
roughly 14× faster training time — the better trade-off for production deployment.

---

# Phase A — Spark DataFrame analytics

## Task 1 — Crime type distribution
*Reema Aldwehi (`raldwehi`)*

DataFrame `groupBy` + descending count on `Primary Type`.

**M1 ↔ M2 — Top 10 (full dataset):**

| Crime type | M1 | M2 |
|------------|---:|---:|
| THEFT | 162,688 | 162,688 |
| BATTERY | 151,930 | 151,930 |
| CRIMINAL DAMAGE | 91,241 | 91,241 |
| NARCOTICS | 74,127 | 74,127 |
| ASSAULT | 54,070 | 54,070 |
| MOTOR VEHICLE THEFT | 48,494 | 48,494 |
| BURGLARY | 39,872 | 39,872 |
| OTHER OFFENSE | 36,893 | 36,893 |
| ROBBERY | 30,991 | 30,991 |
| DECEPTIVE PRACTICE | 30,396 | 30,396 |

Numbers match exactly.

---

## Task 2 — Location hotspots (Spark SQL)
*Lujain Malharbi (`lujainalfaisal`)*

`createOrReplaceTempView` + `spark.sql` for top-10 hotspots.

**M1 ↔ M2 — Top 10:**

| Location | M1 | M2 |
|----------|---:|---:|
| STREET | 245,437 | 248,326 |
| RESIDENCE | 136,238 | 136,393 |
| APARTMENT | 60,925 | 61,235 |
| SIDEWALK | 47,407 | 47,506 |
| OTHER | 29,213 | 29,671 |
| PARKING LOT/GARAGE(NON.RESID.) | 21,876 | 22,436 |
| ALLEY | 18,258 | 18,349 |
| SCHOOL, PUBLIC, BUILDING | 20,516 | 15,776 |
| RESIDENCE-GARAGE | 14,266 | 14,291 |
| SMALL RETAIL STORE | 13,755 | 13,804 |

---

## Task 3 — Year trend
*Leen Althunayan (`L-alth`)*

Yearly counts on the full HDFS dataset. Local matplotlib chart at
`output/yearly_trend.png`.

| Year | Incidents | | Year | Incidents |
|---:|---:|---|---:|---:|
| 2001 | 467,301 | | 2014 | 825 |
| 2002 | 205,266 | | 2015 | 1,105 |
| 2003 | 985 | | 2016 | 1,339 |
| 2023 | 81,461 | | 2025 | 12,710 |

(Full table in the notebook; 2001–2002 dominate, 2023 spike.)

---

## Task 4 — Arrest rate
*Leen Althunayan (`L-alth`)*

**Cluster — overall: 221,932 / 793,073 = 27.98%** (matches M1).

Top arrest rates: NARCOTICS 99.88%, PROSTITUTION 99.88%, LIQUOR LAW 99.83%,
GAMBLING 99.77%, WEAPONS VIOLATION 74.60%, etc. The arrest rate is bimodal —
proactive-policing crimes near 100%, reactive-reporting crimes like THEFT around 14%.

---

# Phase B — MLlib arrest predictor (5% sample)

## Task 5 — Feature pipeline
*Reema Aldwehi (`raldwehi`)*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over
`[crime_code, Hour, dom_code, District]`, 80/20 split with `seed=42`. Vector layout:
`[crime_code, Hour, dom_code, District]`.

## Task 6 — Train and evaluate three classifiers
*Lujain Malharbi (`lujainalfaisal`)*

Cluster results (5% sample of the full HDFS dataset):

| Model | Params | Train (s) | AUC | Accuracy | F1 |
|-------|--------|----------:|----:|---------:|---:|
| Logistic Regression | maxIter=100, regParam=0.01 | 17.9 | 0.6022 | 0.7280 | 0.6376 |
| Random Forest | numTrees=100, maxDepth=5, maxBins=64 | 28.3 | 0.8057 | 0.8156 | 0.7802 |
| **GBT** | maxIter=50, maxDepth=5, maxBins=64 | 401.1 | **0.8241** | **0.8500** | **0.8337** |

**Confusion matrices (TN/FP/FN/TP):**
- LR:  (5549, 93, 2030, 133)
- RF:  (5641, 1, 1438, 725)
- GBT: (5553, 89, 1082, 1081)

**Top model by AUC: GBT (0.8241).**

## Task 7 — Random Forest feature importances
*Leen Althunayan (`L-alth`)*

`crime_code` dominates because the per-crime arrest-rate distribution from Task 4
is itself dominated by crime type. Once a tree splits on the crime label it has
most of its answer. Logistic Regression underperforms tree models because it treats
`crime_code` as a numeric feature and fits a linear coefficient — implying a
meaningless ordering between crime types.

---

# Phase C — Deployment evidence

## Task 9 — Local execution
*Lujain Malharbi (`lujainalfaisal`)*

Notebook executed end-to-end with `jupyter nbconvert --execute` (Python 3.9, PySpark
3.5.1). Section 1 prints:

```
Platform:        local
Spark version:   3.5.1
Spark master:    local[*]
```

All Tasks 1–7 ran; outputs are embedded in `M2_Spark_ML.ipynb`.

## Task 10 — Cluster execution: client mode
*Leen Althunayan (`L-alth`)*

```bash
lsalthunayan@master-node:~$ source /etc/profile.d/hadoop.sh
lsalthunayan@master-node:~$ source /etc/profile.d/spark.sh
lsalthunayan@master-node:~$ spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 768m --executor-cores 1 \
    --driver-memory 1g notebook_runner.py
```

Excerpt from `output/cluster_yarn_log.txt`:

```
Platform:        cluster
Spark version:   3.5.4
Spark master:    yarn
Records ingested: 793,073
Phase B working set: 39,534 rows  (5% sample, seed=42)
Train rows: 31,728 | Test rows: 7,806

>> LogisticRegression  AUC 0.6022, Train 17.9s
>> RandomForest        AUC 0.8057, Train 28.3s
>> GBT                 AUC 0.8241, Train 401.1s
Top model by AUC: GBT (0.8241)
```

YARN application: `application_1778738889964_0119`.

## Task 11 — spark-submit (cluster mode)
*Reema Aldwehi (`raldwehi`)*

Per the May 2026 spec update, Task 11 uses `--deploy-mode cluster`:

```bash
raldwehi@master-node:~$ spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```

YARN application: `application_1778738889964_0120` — `final status: SUCCEEDED`.

Application stdout in `output/spark_submit/run.log`. The console.log
(`output/spark_submit/console.log`) captures the spark-submit invocation and YARN's
progress reports.

---

## Spec note — executor cores

The M2 spec lists `--executor-cores 2`. The course YARN cluster's maximum container
allocation is `<memory:1536, vCores:1>` — requesting 2 vcores returns
`InvalidResourceRequestException`. We therefore use `--executor-cores 1`, the same
setting M1 used.

---

## Member contributions

| Member | Tasks | Contribution |
|--------|-------|--------------|
| Reema Aldwehi (`raldwehi`) | 1, 5, 11 | Crime-type DataFrame query; feature pipeline; spark-submit cluster mode submission |
| Lujain Malharbi (`lujainalfaisal`) | 2, 6, 9 | Spark SQL hotspots; three-classifier comparison; local notebook execution evidence |
| Leen Althunayan (`L-alth`) | 3, 4, 7, 10 | Year trend + chart; arrest-rate analysis; RF feature importances; yarn-client cluster execution evidence |

## How to reproduce

Locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Spark_ML.ipynb --output M2_Spark_ML.ipynb
```

On the cluster:
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```
