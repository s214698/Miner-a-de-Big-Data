# Databricks notebook source
# MAGIC %pip install huggingface_hub 
# MAGIC from huggingface_hub import hf_hub_download
# MAGIC from pyspark.sql import functions as F
# MAGIC
# MAGIC hf_hub_download(
# MAGIC     repo_id="McAuley-Lab/Amazon-Reviews-2023",
# MAGIC     repo_type="dataset",
# MAGIC     filename="raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl",
# MAGIC     local_dir="/Volumes/workspace/default/raw",
# MAGIC )

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

schema = StructType([
    StructField("parent_asin", StringType()),
    StructField("title", StringType()),
    StructField("store", StringType()),
    StructField("price", StringType()),            # read as text, because it can be null or a non-number
    StructField("categories", ArrayType(StringType())),
    StructField("features", ArrayType(StringType())),
    StructField("description", ArrayType(StringType())),
])

meta_raw = spark.read.schema(schema).json(
    "/Volumes/workspace/default/raw/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl")

meta_raw.show(5, truncate=60)

# COMMAND ----------

train = spark.table("workspace.default.amazon_train_nonfilter")
val   = spark.table("workspace.default.amazon_val_nonfilter")
test  = spark.table("workspace.default.amazon_test_nonfilter")
train.printSchema()

# COMMAND ----------

meta_raw = meta_raw.withColumnRenamed("parent_asin", "product_id")
meta_raw.show(5, truncate=60)


# COMMAND ----------

from pyspark.sql import functions as F

txt = F.concat_ws(" ",
    F.coalesce(F.col("title"), F.lit("")),
    F.coalesce(F.concat_ws(" ", "categories"), F.lit("")),
    F.coalesce(F.col("store"), F.lit("")),
    F.coalesce(F.concat_ws(" ", "features"), F.lit("")),
    F.coalesce(F.concat_ws(" ", "description"), F.lit("")))

meta = (meta_raw
        .withColumn("text", txt)
        .select(F.col("product_id"),
                F.expr("try_cast(price as double)").alias("price"),
                "text")
        .dropDuplicates(["product_id"]))

meta.show(5, truncate=80)

# COMMAND ----------

# MAGIC %md
# MAGIC All items have text 

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

im = (meta
      .withColumn("text", F.coalesce("text", F.lit("")))
      .withColumn("has_price", F.when(F.col("price").isNotNull(), 1.0).otherwise(0.0))
      .withColumn("log_price", F.log1p(F.coalesce("price", F.lit(0.0)))))

tok = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+", minTokenLength=2).transform(im)
clean = StopWordsRemover(inputCol="words", outputCol="clean").transform(tok)
cv = CountVectorizer(inputCol="clean", outputCol="tf", vocabSize=20000, minDF=5).fit(clean)
tf = cv.transform(clean)
tfidf = IDF(inputCol="tf", outputCol="tfidf").fit(tf).transform(tf)

feat = (VectorAssembler(inputCols=["tfidf", "log_price", "has_price"], outputCol="features")
        .transform(tfidf).select("product_id", "features"))
feat.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_features_nofilter")
feat = spark.table("workspace.default.item_features_nofilter")

# COMMAND ----------

mu = train.agg(F.avg("rating")).first()[0]

item_b = train.groupBy("product_id").agg(
    (F.sum(F.col("rating") - mu) / (F.count("*") + 10.0)).alias("b_i"))

user_b = (train.join(item_b, "product_id")
          .groupBy("user_id").agg(
    (F.sum(F.col("rating") - mu - F.col("b_i")) / (F.count("*") + 5.0)).alias("b_u")))

# COMMAND ----------

n_tr = train.groupBy("product_id").count()
labels = (item_b.join(n_tr, "product_id").filter("count >= 5")
          .join(feat, "product_id").select("product_id", "features", F.col("b_i").alias("label")))

fit_set, hold = labels.randomSplit([0.8, 0.2], seed=42)
ev = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")

for reg in [0.01, 0.1, 1.0, 10.0]:
    m = LinearRegression(featuresCol="features", labelCol="label",
                         regParam=reg, elasticNetParam=0.0).fit(fit_set)
    print(reg, round(ev.evaluate(m.transform(hold)), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC 0.01 0.3519
# MAGIC 0.1 0.3371
# MAGIC 1.0 0.3348
# MAGIC 10.0 0.3508

# COMMAND ----------

hold.agg(F.stddev("label")).show()

# COMMAND ----------

best = LinearRegression(featuresCol="features", labelCol="label",
                        regParam=0.5, elasticNetParam=0.0).fit(labels)
pred_b = best.transform(feat).select("product_id", F.col("prediction").alias("b_text"))

clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))
rm = lambda d, col: RegressionEvaluator(metricName="rmse", labelCol="rating",
                                        predictionCol=col).evaluate(d)

v = (val.join(item_b, "product_id", "left")      # b_i is null for new items
        .join(user_b, "user_id", "left")
        .join(pred_b, "product_id", "left")
        .withColumn("b_u", F.coalesce("b_u", F.lit(0.0)))
        .withColumn("p_base", clip(F.lit(mu) + F.col("b_u") + F.coalesce("b_i", F.lit(0.0))))
        .withColumn("p_text", clip(F.lit(mu) + F.col("b_u") +
                    F.coalesce("b_i", F.coalesce("b_text", F.lit(0.0))))))

new = v.filter(F.col("b_i").isNull())
print("new-item rows:", new.count())
print("baseline:", round(rm(new, "p_base"), 4), "| with text:", round(rm(new, "p_text"), 4))
print("all rows: ", round(rm(v, "p_base"), 4), "| with text:", round(rm(v, "p_text"), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC This is only used for completely new items.

# COMMAND ----------

folds = labels.withColumn("fold", F.abs(F.hash("product_id")) % 5)

def fit_text(d):
    return LinearRegression(featuresCol="features", labelCol="label",
                            regParam=0.5, elasticNetParam=0.0).fit(d)

# out-of-fold predictions for items that were used as labels
oof = None
for k in range(5):
    m = fit_text(folds.filter(F.col("fold") != k))
    p = (m.transform(folds.filter(F.col("fold") == k))
          .select("product_id", F.col("prediction").alias("b_text")))
    oof = p if oof is None else oof.unionByName(p)

# all other items (fewer than 5 train ratings, or new) were never in the training set
full = fit_text(labels)
rest = (full.transform(feat.join(labels.select("product_id"), "product_id", "left_anti"))
            .select("product_id", F.col("prediction").alias("b_text")))

b_text_all = oof.unionByName(rest)
b_text_all.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_b_text_2")
b_text_all = spark.table("workspace.default.item_b_text_2")

# COMMAND ----------

item_stats = train.groupBy("product_id").agg(F.sum(F.col("rating") - mu).alias("s"),
                                           F.count("*").alias("n"))

def make_item_b2(lam_i):
    return (b_text_all.join(item_stats, "product_id", "left")
            .fillna({"s": 0.0, "n": 0})
            .withColumn("b_i2", (F.col("s") + lam_i * F.col("b_text")) / (F.col("n") + lam_i))
            .select("product_id", "b_i2"))

for lam_i in [5, 10, 20, 30]:
    ib = make_item_b2(lam_i)
    v = (val.join(ib, "product_id", "left").join(user_b, "user_id", "left")
            .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) + F.coalesce("b_i2", F.lit(0.0)))))
    print(lam_i, "all rows:", round(rm(v, "p"), 4))

# COMMAND ----------

for lam_i in [5, 10, 20, 50]:
    for use_text in [True, False]:
        ib = (b_text_all.join(item_stats, "product_id", "left")
              .fillna({"s": 0.0, "n": 0})
              .withColumn("prior", F.col("b_text") if use_text else F.lit(0.0))
              .withColumn("b_i2", (F.col("s") + lam_i * F.col("prior")) / (F.col("n") + lam_i))
              .select("product_id", "b_i2"))
        v = (val.join(ib, "product_id", "left").join(user_b, "user_id", "left")
                .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) + F.coalesce("b_i2", F.lit(0.0)))))
        print(lam_i, "text" if use_text else "zero", round(rm(v, "p"), 4))

# COMMAND ----------

def predict_baseline(d):
    return (d.join(item_b, "product_id", "left").join(user_b, "user_id", "left")
             .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0))
                                   + F.coalesce("b_i", F.lit(0.0)))))

for name, d in [("val", val), ("test", test)]:
    pred = predict_baseline(d)
    print(f"{name}: rows={d.count()} | RMSE={rm(pred, 'p'):.5f}")

# COMMAND ----------

def make_item_b_text(lam_text=10):
    return (b_text_all.join(item_stats, "product_id", "left")
            .fillna({"s": 0.0, "n": 0})
            .withColumn("b_i2", (F.col("s") + lam_text * F.col("b_text")) / (F.col("n") + lam_text))
            .select("product_id", "b_i2"))

def predict_text(d, ib):
    return (d.join(ib, "product_id", "left").join(user_b, "user_id", "left")
             .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0))
                                   + F.coalesce("b_i2", F.lit(0.0)))))

text_ib = make_item_b_text(10)

for name, d in [("val", val), ("test", test)]:
    pred = predict_text(d, text_ib)
    print(f"{name}: rows={d.count()} | RMSE={rm(pred, 'p'):.5f}")

# COMMAND ----------

def make_item_b(lam_i, use_text):
    return (b_text_all.join(item_stats, "product_id", "left")
            .fillna({"s": 0.0, "n": 0})
            .withColumn("prior", F.col("b_text") if use_text else F.lit(0.0))
            .withColumn("b_i2", (F.col("s") + lam_i * F.col("prior")) / (F.col("n") + lam_i))
            .select("product_id", "b_i2"))

def predict(d, ib):
    return (d.join(ib, "product_id", "left").join(user_b, "user_id", "left")
             .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0))
                                   + F.coalesce("b_i2", F.lit(0.0)))))

for name, d in [("val", val), ("test", test)]:
    base = predict(d, make_item_b(10, False))   # tuned baseline, zero prior
    text = predict(d, make_item_b(20, True))    # baseline + text prior
    print(name, "rows:", d.count(),
          "| baseline:", round(rm(base, "p"), 4),
          "| with text:", round(rm(text, "p"), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC val rows: 665176 | baseline: 1.2459 | with text: 1.2428
# MAGIC test rows: 663052 | baseline: 1.1549 | with text: 1.1531