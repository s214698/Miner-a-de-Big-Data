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

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
test  = spark.table("workspace.default.amazon_test")
train.printSchema()

# COMMAND ----------

item_map = (train.select("product_id", "item_idx")
            .unionByName(val.select("product_id", "item_idx"))
            .unionByName(test.select("product_id", "item_idx"))
            .distinct())
print(item_map.count())

# COMMAND ----------

meta_raw = meta_raw.withColumnRenamed("parent_asin", "product_id")
meta_raw.show(5, truncate=60)


# COMMAND ----------

item_meta = item_map.join(meta_raw, "product_id", "left")   # left: keep items with no metadata
item_meta.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_meta")
item_meta = spark.table("workspace.default.item_meta")

# COMMAND ----------

item_meta.show(5, truncate=60)

# COMMAND ----------

from pyspark.sql import functions as F

txt = F.concat_ws(" ",
    F.coalesce(F.col("title"), F.lit("")),
    F.coalesce(F.concat_ws(" ", "categories"), F.lit("")),
    F.coalesce(F.col("store"), F.lit("")),
    F.coalesce(F.concat_ws(" ", "features"), F.lit("")),
    F.coalesce(F.concat_ws(" ", "description"), F.lit("")))

meta = (item_meta
        .withColumn("text", txt)
        .select(F.col("product_id"),
                "item_idx",
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
        .transform(tfidf).select("item_idx", "features"))
feat.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_features")
feat = spark.table("workspace.default.item_features")

# COMMAND ----------

mu = train.agg(F.avg("rating")).first()[0]
lam = 5   # shrinkage: biases from few ratings are pulled toward 0

item_b = train.groupBy("item_idx").agg(
    (F.sum(F.col("rating") - mu) / (F.count("*") + lam)).alias("b_i"))

user_b = (train.join(item_b, "item_idx")
          .groupBy("user_idx").agg(
    (F.sum(F.col("rating") - mu - F.col("b_i")) / (F.count("*") + lam)).alias("b_u")))

# COMMAND ----------

n_tr = train.groupBy("item_idx").count()
labels = (item_b.join(n_tr, "item_idx").filter("count >= 5")
          .join(feat, "item_idx").select("item_idx", "features", F.col("b_i").alias("label")))

fit_set, hold = labels.randomSplit([0.8, 0.2], seed=42)
ev = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")

for reg in [0.01, 0.1, 1.0, 10.0]:
    m = LinearRegression(featuresCol="features", labelCol="label",
                         regParam=reg, elasticNetParam=0.0).fit(fit_set)
    print(reg, round(ev.evaluate(m.transform(hold)), 4))

# COMMAND ----------

hold.agg(F.stddev("label")).show()

# COMMAND ----------

best = LinearRegression(featuresCol="features", labelCol="label",
                        regParam=1.0, elasticNetParam=0.0).fit(labels)
pred_b = best.transform(feat).select("item_idx", F.col("prediction").alias("b_text"))

clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))
rm = lambda d, col: RegressionEvaluator(metricName="rmse", labelCol="rating",
                                        predictionCol=col).evaluate(d)

v = (val.join(item_b, "item_idx", "left")      
        .join(user_b, "user_idx", "left")
        .join(pred_b, "item_idx", "left")
        .withColumn("b_u", F.coalesce("b_u", F.lit(0.0)))
        .withColumn("p_base", clip(F.lit(mu) + F.col("b_u") + F.coalesce("b_i", F.lit(0.0))))
        .withColumn("p_text", clip(F.lit(mu) + F.col("b_u") +
                    F.coalesce("b_i", F.coalesce("b_text", F.lit(0.0))))))

new = v.filter(F.col("b_i").isNull())
print("new-item rows:", new.count())
print("baseline:", round(rm(new, "p_base"), 4), "| with text:", round(rm(new, "p_text"), 4))
print("all rows: ", round(rm(v, "p_base"), 4), "| with text:", round(rm(v, "p_text"), 4))

# COMMAND ----------

folds = labels.withColumn("fold", F.abs(F.hash("item_idx")) % 5)

def fit_text(d):
    return LinearRegression(featuresCol="features", labelCol="label",
                            regParam=1.0, elasticNetParam=0.0).fit(d)

# out-of-fold predictions for items that were used as labels
oof = None
for k in range(5):
    m = fit_text(folds.filter(F.col("fold") != k))
    p = (m.transform(folds.filter(F.col("fold") == k))
          .select("item_idx", F.col("prediction").alias("b_text")))
    oof = p if oof is None else oof.unionByName(p)

# all other items (fewer than 5 train ratings, or new) were never in the training set
full = fit_text(labels)
rest = (full.transform(feat.join(labels.select("item_idx"), "item_idx", "left_anti"))
            .select("item_idx", F.col("prediction").alias("b_text")))

b_text_all = oof.unionByName(rest)
b_text_all.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_b_text")
b_text_all = spark.table("workspace.default.item_b_text")

# COMMAND ----------

item_stats = train.groupBy("item_idx").agg(F.sum(F.col("rating") - mu).alias("s"),
                                           F.count("*").alias("n"))

def make_item_b2(lam_i):
    return (b_text_all.join(item_stats, "item_idx", "left")
            .fillna({"s": 0.0, "n": 0})
            .withColumn("b_i2", (F.col("s") + lam_i * F.col("b_text")) / (F.col("n") + lam_i))
            .select("item_idx", "b_i2"))

for lam_i in [2, 5, 10, 20]:
    ib = make_item_b2(lam_i)
    v = (val.join(ib, "item_idx", "left").join(user_b, "user_idx", "left")
            .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) + F.coalesce("b_i2", F.lit(0.0)))))
    print(lam_i, "all rows:", round(rm(v, "p"), 4))

# COMMAND ----------

for lam_i in [5, 10, 20, 50]:
    for use_text in [True, False]:
        ib = (b_text_all.join(item_stats, "item_idx", "left")
              .fillna({"s": 0.0, "n": 0})
              .withColumn("prior", F.col("b_text") if use_text else F.lit(0.0))
              .withColumn("b_i2", (F.col("s") + lam_i * F.col("prior")) / (F.col("n") + lam_i))
              .select("item_idx", "b_i2"))
        v = (val.join(ib, "item_idx", "left").join(user_b, "user_idx", "left")
                .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) + F.coalesce("b_i2", F.lit(0.0)))))
        print(lam_i, "text" if use_text else "zero", round(rm(v, "p"), 4))

# COMMAND ----------

def make_item_b(lam_i, use_text):
    return (b_text_all.join(item_stats, "item_idx", "left")
            .fillna({"s": 0.0, "n": 0})
            .withColumn("prior", F.col("b_text") if use_text else F.lit(0.0))
            .withColumn("b_i2", (F.col("s") + lam_i * F.col("prior")) / (F.col("n") + lam_i))
            .select("item_idx", "b_i2"))

def predict(d, ib):
    return (d.join(ib, "item_idx", "left").join(user_b, "user_idx", "left")
             .withColumn("p", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0))
                                   + F.coalesce("b_i2", F.lit(0.0)))))

for name, d in [("val", val), ("test", test)]:
    base = predict(d, make_item_b(10, False))   # tuned baseline, zero prior
    text = predict(d, make_item_b(20, True))    # baseline + text prior
    print(name, "rows:", d.count(),
          "| baseline:", round(rm(base, "p"), 4),
          "| with text:", round(rm(text, "p"), 4))