# Databricks notebook source
# MAGIC %md
# MAGIC # Prueba de significancia: baseline vs. baseline + prior de texto
# MAGIC
# MAGIC Prueba **pareada** sobre el conjunto de prueba. Para cada fila k:
# MAGIC
# MAGIC `d_k = (r_k - p_base)^2 - (r_k - p_texto)^2`
# MAGIC
# MAGIC - `mean(d)` = reducción del MSE; `SE = sd(d) / sqrt(n)`; `z = mean(d) / SE`.
# MAGIC - También se calcula un SE agrupado por usuario (las filas de un mismo usuario no son independientes)
# MAGIC   y una prueba de signos sobre las filas donde la predicción cambia.
# MAGIC
# MAGIC Requiere haber ejecutado antes los notebooks de preprocesamiento y de contenido
# MAGIC (tablas `item_b_text` para 5-core e `item_b_text_2` para el conjunto completo).

# COMMAND ----------

import math
from pyspark.sql import functions as F

LAM_I, LAM_U = 10.0, 5.0   # baseline tuned on validation

CONFIGS = {
    "5-core": dict(train="workspace.default.amazon_train",
                   test="workspace.default.amazon_test",
                   b_text="workspace.default.item_b_text",
                   item="item_idx", user="user_idx", lam_text=20.0),
    "completo": dict(train="workspace.default.amazon_train_nonfilter",
                     test="workspace.default.amazon_test_nonfilter",
                     b_text="workspace.default.item_b_text_2",
                     item="product_id", user="user_id", lam_text=10.0),
}

clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))

# COMMAND ----------

def predictions(cfg):
    """Baseline and text-prior predictions on the test set, same rows for both."""
    train, test = spark.table(cfg["train"]), spark.table(cfg["test"])
    b_text = spark.table(cfg["b_text"])
    it, us, lam = cfg["item"], cfg["user"], cfg["lam_text"]

    mu = train.agg(F.avg("rating")).first()[0]
    stats = train.groupBy(it).agg(F.sum(F.col("rating") - mu).alias("s"),
                                  F.count("*").alias("n"))

    # baseline: item bias shrunk toward 0, user bias on the residuals
    b_i = stats.select(it, (F.col("s") / (F.col("n") + LAM_I)).alias("b_i"))
    b_u = (train.join(b_i, it).groupBy(us)
           .agg((F.sum(F.col("rating") - mu - F.col("b_i")) /
                 (F.count("*") + LAM_U)).alias("b_u")))

    # text prior: item bias shrunk toward b_text instead of 0
    b_i_text = (b_text.join(stats, it, "left").fillna({"s": 0.0, "n": 0})
                .select(it, ((F.col("s") + lam * F.col("b_text")) /
                             (F.col("n") + lam)).alias("b_i_text")))

    seen = stats.select(it, F.lit(1).alias("item_seen"))

    return (test.select(us, it, "rating")
            .join(b_i, it, "left").join(b_i_text, it, "left")
            .join(b_u, us, "left").join(seen, it, "left")
            .fillna({"b_i": 0.0, "b_i_text": 0.0, "b_u": 0.0, "item_seen": 0})
            .withColumn("p_base", clip(F.lit(mu) + F.col("b_u") + F.col("b_i")))
            .withColumn("p_text", clip(F.lit(mu) + F.col("b_u") + F.col("b_i_text")))
            .withColumn("se_base", (F.col("rating") - F.col("p_base")) ** 2)
            .withColumn("se_text", (F.col("rating") - F.col("p_text")) ** 2)
            .withColumn("d", F.col("se_base") - F.col("se_text")))


def p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def paired_test(df, user_col):
    s = df.agg(F.count("*").alias("n"),
               F.avg("d").alias("mean_d"),
               F.stddev("d").alias("sd_d"),
               F.avg("se_base").alias("mse_base"),
               F.avg("se_text").alias("mse_text"),
               F.sum(F.when(F.col("d") > 0, 1).otherwise(0)).alias("better"),
               F.sum(F.when(F.col("d") < 0, 1).otherwise(0)).alias("worse")).first()
    n, md, sd = s["n"], s["mean_d"], s["sd_d"]

    # naive (row-independent) standard error
    se = sd / math.sqrt(n)

    # cluster-robust standard error by user
    per_user = df.groupBy(user_col).agg(F.sum("d").alias("D"), F.count("*").alias("m"))
    v = per_user.agg(F.sum((F.col("D") - F.col("m") * F.lit(md)) ** 2).alias("v"),
                     F.count("*").alias("G")).first()
    G = v["G"]
    se_cl = math.sqrt(v["v"] * G / (G - 1)) / n

    rmse_b, rmse_t = math.sqrt(s["mse_base"]), math.sqrt(s["mse_text"])
    lo, hi = md - 1.96 * se_cl, md + 1.96 * se_cl          # CI for ΔMSE (clustered)
    d_rmse = lambda dm: rmse_b - math.sqrt(s["mse_base"] - dm)

    # sign test on rows where the prediction changed
    k, m = s["better"], s["better"] + s["worse"]
    z_sign = (k - m / 2) / math.sqrt(m / 4) if m > 0 else float("nan")

    return dict(n=n, rmse_base=rmse_b, rmse_text=rmse_t,
                d_rmse=rmse_b - rmse_t, ci_low=d_rmse(lo), ci_high=d_rmse(hi),
                d_mse=md, sd_d=sd,
                z=md / se, p=p_two_sided(md / se),
                z_cluster=md / se_cl, p_cluster=p_two_sided(md / se_cl),
                rows_better=k, rows_worse=s["worse"],
                z_sign=z_sign, p_sign=p_two_sided(z_sign))

# COMMAND ----------

results = []
for name, cfg in CONFIGS.items():
    df = predictions(cfg).cache()
    r = paired_test(df, cfg["user"])
    results.append((name, "todas las filas", r))
    for flag, label in [(1, "ítem visto"), (0, "ítem nuevo")]:
        sub = df.filter(F.col("item_seen") == flag)
        if sub.limit(1).count():
            results.append((name, label, paired_test(sub, cfg["user"])))
    df.unpersist()

for name, group, r in results:
    print(f"\n== {name} | {group} ==")
    print(f"n = {r['n']:,}")
    print(f"RMSE baseline = {r['rmse_base']:.5f} | RMSE texto = {r['rmse_text']:.5f}")
    print(f"ΔRMSE = {r['d_rmse']:.5f}  IC 95 % (agrupado por usuario) = "
          f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}]")
    print(f"ΔMSE = {r['d_mse']:.5f} | sd(d) = {r['sd_d']:.4f}")
    print(f"z = {r['z']:.1f}, p = {r['p']:.2e} | z agrupado = {r['z_cluster']:.1f}, "
          f"p agrupado = {r['p_cluster']:.2e}")
    print(f"filas que mejoran / empeoran = {r['rows_better']:,} / {r['rows_worse']:,} "
          f"(prueba de signos z = {r['z_sign']:.1f}, p = {r['p_sign']:.2e})")

# COMMAND ----------

rows = [(name, group, *[float(r[k]) for k in
         ["n", "rmse_base", "rmse_text", "d_rmse", "ci_low", "ci_high", "z", "p",
          "z_cluster", "p_cluster", "rows_better", "rows_worse", "p_sign"]])
        for name, group, r in results]
cols = ["dataset", "group", "n", "rmse_base", "rmse_text", "d_rmse", "ci_low", "ci_high",
        "z", "p", "z_cluster", "p_cluster", "rows_better", "rows_worse", "p_sign"]
(spark.createDataFrame(rows, cols)
      .write.format("delta").mode("overwrite")
      .saveAsTable("workspace.default.significance_results"))
spark.table("workspace.default.significance_results").show(truncate=False)
