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
meta.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_meta2")

# COMMAND ----------

# MAGIC %md
# MAGIC All items have text 
# MAGIC
# MAGIC **Now for training**

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

meta = spark.table("workspace.default.item_meta2")
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

# COMMAND ----------

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
test  = spark.table("workspace.default.amazon_test")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")
feat = spark.table("workspace.default.item_features")
mu = train.agg(F.avg("rating")).first()[0]

def add_base(d):
    return (d.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
             .fillna({"b_i": 0.0, "b_u": 0.0})
             .withColumn("base", F.lit(mu) + F.col("b_i") + F.col("b_u")))

train_b = add_base(train).withColumn("resid", F.col("rating") - F.col("base"))
val_b   = add_base(val)

n_tr = train.groupBy("item_idx").count()
labels = (item_b.join(n_tr, "item_idx").filter("count >= 5")
          .join(feat, "item_idx").select("item_idx", "features", F.col("b_i").alias("label")))

fit_set, hold = labels.randomSplit([0.8, 0.2], seed=42)
ev = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")

for reg in [0.01, 0.1, 1.0, 10.0]:
    m = LinearRegression(featuresCol="features", labelCol="label",
                         regParam=reg, elasticNetParam=0.0).fit(fit_set)
    print(reg, round(ev.evaluate(m.transform(hold)), 4))

hold.agg(F.stddev("label")).show()

# COMMAND ----------

hold.agg(F.stddev("label")).show()

# COMMAND ----------

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
test  = spark.table("workspace.default.amazon_test")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")
feat = spark.table("workspace.default.item_features")
mu = train.agg(F.avg("rating")).first()[0]

def add_base(d):
    return (d.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
             .fillna({"b_i": 0.0, "b_u": 0.0})
             .withColumn("base", F.lit(mu) + F.col("b_i") + F.col("b_u")))

train_b = add_base(train).withColumn("resid", F.col("rating") - F.col("base"))
val_b   = add_base(val)

n_tr = train.groupBy("item_idx").count()
labels = (item_b.join(n_tr, "item_idx").filter("count >= 5")
          .join(feat, "item_idx").select("item_idx", "features", F.col("b_i").alias("label")))

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

from pyspark.sql import functions as F
train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
test  = spark.table("workspace.default.amazon_test")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")
feat  = spark.table("workspace.default.item_features")
mu = train.agg(F.avg("rating")).first()[0]

def add_base(d):
    return (d.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
             .fillna({"b_i": 0.0, "b_u": 0.0})
             .withColumn("base", F.lit(mu) + F.col("b_i") + F.col("b_u")))

train_b = add_base(train).withColumn("resid", F.col("rating") - F.col("base"))
val_b = add_base(val)

n_tr = train.groupBy("item_idx").count()
labels = (item_b.join(n_tr, "item_idx").filter("count >= 5")
          .join(feat, "item_idx").select("item_idx", "features", F.col("b_i").alias("label")))
          # Save labels
labels.write.mode("overwrite").saveAsTable("workspace.default.item_labels")

# COMMAND ----------

clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))
rm = lambda d, col: RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol=col).evaluate(d)

reg_values = [0.1, 0.5, 1.0, 3.0]
results = []

for reg in reg_values:
    model = LinearRegression(featuresCol="features", labelCol="label", regParam=reg, elasticNetParam=0.0).fit(labels)
    pred_b = model.transform(feat).select("item_idx", F.col("prediction").alias("b_text"))

    v = (val.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left").join(pred_b, "item_idx", "left")
           .withColumn("b_u", F.coalesce("b_u", F.lit(0.0)))
           .withColumn("p_base", clip(F.lit(mu) + F.col("b_u") + F.coalesce("b_i", F.lit(0.0))))
           .withColumn("p_text", clip(F.lit(mu) + F.col("b_u") + F.coalesce("b_i", F.coalesce("b_text", F.lit(0.0))))))

    new = v.filter(F.col("b_i").isNull())
    results.append((reg, rm(new, "p_base"), rm(new, "p_text"), rm(v, "p_base"), rm(v, "p_text")))

results_df = spark.createDataFrame(results, ["regParam", "new_baseline_rmse", "new_text_rmse", "all_baseline_rmse", "all_text_rmse"])

results_df.orderBy("regParam").show(truncate=False)
results_df.write.mode("overwrite").saveAsTable("workspace.default.amazon_reg_validation")

# Best regularization value based on new-item validation RMSE
results_df.orderBy("new_text_rmse").show(truncate=False)

# COMMAND ----------

best = LinearRegression(featuresCol="features", labelCol="label", regParam=0.5, elasticNetParam=0.0).fit(labels)
w = best.coefficients.toArray()[:len(cv.vocabulary)]   # first block = TF-IDF weights, in vocabulary order
ranked = sorted(zip(cv.vocabulary, w), key=lambda t: t[1])
print("most negative:", ranked[:15])
print("most positive:", ranked[-15:])

# COMMAND ----------

# MAGIC %md
# MAGIC The results above are for the item text prior only counting when there the item in val is new. Now we try to let it count with regualizer.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

labels = spark.table("workspace.default.item_labels")
folds = labels.withColumn("fold", F.abs(F.hash("item_idx")) % 5)

def fit_text(d):
    return LinearRegression(featuresCol="features", labelCol="label",
                            regParam=0.5, elasticNetParam=0.0).fit(d)

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

from pyspark.sql import functions as F
train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
test  = spark.table("workspace.default.amazon_test")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")
mu = train.agg(F.avg("rating")).first()[0]

def add_base(d):
    return (d.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
             .fillna({"b_i": 0.0, "b_u": 0.0})
             .withColumn("base", F.lit(mu) + F.col("b_i") + F.col("b_u")))

rm = lambda d, col: RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol=col).evaluate(d)
clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))
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

# Baseline on test
test_base = add_base(test).withColumn("p_base", clip(F.col("base")))
print("Baseline test RMSE:", round(rm(test_base, "p_base"), 4))

# Final content-enhanced model on test (lambda = 20)
ib = make_item_b2(20)

test_final = (test.join(ib, "item_idx", "left").join(user_b, "user_idx", "left")
                .withColumn("p_final", clip(F.lit(mu) + F.coalesce("b_u", F.lit(0.0)) +
                                             F.coalesce("b_i2", F.lit(0.0)))))

print("Final model test RMSE:", round(rm(test_final, "p_final"), 4))

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