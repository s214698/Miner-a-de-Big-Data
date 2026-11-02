# Databricks notebook source
# DBTITLE 1,Load train/val/test
train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
test  = spark.table("workspace.default.amazon_test")

print(train.count(), val.count(), test.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Tune baseline model:

# COMMAND ----------

from pyspark.sql import functions as F

mu = train.agg(F.avg("rating")).first()[0]
clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))

# sufficient statistics per item, computed once
item_s = train.groupBy("item_idx").agg(F.sum(F.col("rating") - mu).alias("s_i"),
                                       F.count("*").alias("n_i"))

def rmse(d):
    return d.agg(F.sqrt(F.avg((F.col("rating") - F.col("p"))**2))).first()[0]

results = []
for lam_i in [2, 5, 10, 20]:
    item_b = item_s.select("item_idx", (F.col("s_i") / (F.col("n_i") + lam_i)).alias("b_i"))
    # user statistics depend on lam_i (through b_i), but not on lam_u
    user_s = (train.join(item_b, "item_idx").groupBy("user_idx")
              .agg(F.sum(F.col("rating") - mu - F.col("b_i")).alias("s_u"),
                   F.count("*").alias("n_u")))
    for lam_u in [1, 2, 5, 10]:
        user_b = user_s.select("user_idx", (F.col("s_u") / (F.col("n_u") + lam_u)).alias("b_u"))
        v = (val.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
                .fillna({"b_i": 0.0, "b_u": 0.0})
                .withColumn("p", clip(F.lit(mu) + F.col("b_i") + F.col("b_u"))))
        r = float(rmse(v))
        results.append((lam_i, float(lam_u), r))
        print(lam_i, lam_u, round(r, 4))

best_i, best_u, best_r = min(results, key=lambda x: x[2])
print("best: lam_i =", best_i, "| lam_u =", best_u, "| val RMSE =", round(best_r, 4))

# COMMAND ----------

lam_i, lam_u = best_i, best_u

item_b = item_s.select("item_idx", (F.col("s_i") / (F.col("n_i") + lam_i)).alias("b_i"))
user_b = (train.join(item_b, "item_idx").groupBy("user_idx")
          .agg((F.sum(F.col("rating") - mu - F.col("b_i")) / (F.count("*") + lam_u)).alias("b_u")))

item_b.write.format("delta").mode("overwrite").saveAsTable("workspace.default.final_item_b")
user_b.write.format("delta").mode("overwrite").saveAsTable("workspace.default.final_user_b")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")

def add_base(d):
    return (d.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
             .fillna({"b_i": 0.0, "b_u": 0.0})
             .withColumn("base", F.lit(mu) + F.col("b_i") + F.col("b_u")))

train_b = add_base(train).withColumn("resid", F.col("rating") - F.col("base"))
val_b   = add_base(val)

# COMMAND ----------

# MAGIC %md
# MAGIC The hyperparameters have been set acording to the RMSE on validationset resulting in lam_i = 10 | lam_u = 5.0 | val RMSE = 1.2454. 
# MAGIC
# MAGIC Now we wish to tune the ASL
# MAGIC
# MAGIC Now We try to train on residual in stead of the raw ratings:

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

ev = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="final")
clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))

results = []
for rank in [16, 32, 64]:
    for reg in [0.05, 0.1, 0.3, 1.0]:
        m = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="resid",
                rank=rank, regParam=reg, maxIter=15,
                coldStartStrategy="drop", seed=42).fit(train_b)
        p = m.transform(val_b).withColumn("final", clip(F.col("base") + F.col("prediction")))
        r = ev.evaluate(p)
        r_base = ev.evaluate(p.withColumn("final", clip(F.col("base")))) 
        results.append((rank, reg, r, r_base))
        print(rank, reg, round(r, 4), "baseline:", round(r_base, 4))

spark.createDataFrame(results, ["rank", "regParam", "val_residuals_rmse", "val_baseline"]) \
     .write.mode("overwrite").saveAsTable("workspace.default.als_tuning2")

# COMMAND ----------

results2 = []
for rank in [75]:
    for reg in [0.3]:
        m = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="resid",
                rank=rank, regParam=reg, maxIter=15,
                coldStartStrategy="drop", seed=42).fit(train_b)
        p = m.transform(val_b).withColumn("final", clip(F.col("base") + F.col("prediction")))
        r = ev.evaluate(p)
        r_base = ev.evaluate(p.withColumn("final", clip(F.col("base")))) 
        results2.append((rank, reg, r, r_base))
        print(rank, reg, round(r, 4), "baseline:", round(r_base, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC Collabourative filter

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")


tr = (train.join(item_b, "item_idx").join(user_b, "user_idx")
      .withColumn("e", F.col("rating") - (F.lit(mu) + F.col("b_i") + F.col("b_u")))
      .select("user_idx", "item_idx", "e", "timestamp"))

w = Window.partitionBy("user_idx").orderBy(F.col("timestamp").desc())
tr = tr.withColumn("rn", F.row_number().over(w)).filter("rn <= 100").drop("rn")

a = tr.select("user_idx", F.col("item_idx").alias("i"), F.col("e").alias("ei"))
b = tr.select("user_idx", F.col("item_idx").alias("j"), F.col("e").alias("ej"))

lam2 = 20
sims = (a.join(b, "user_idx").filter("i < j")        
        .groupBy("i", "j")
        .agg(F.count("*").alias("n"),
             F.sum(F.col("ei") * F.col("ej")).alias("dot"),
             F.sum(F.col("ei")**2).alias("ni"),
             F.sum(F.col("ej")**2).alias("nj"))
        .filter("n >= 5")   #Experiment with 2,3,4                       
        .withColumn("sim", F.col("dot") / F.sqrt(F.col("ni") * F.col("nj"))
                           * F.col("n") / (F.col("n") + lam2))   
        .select("i", "j", "sim"))

# make it symmetric, keep the 50 strongest neighbors per item
sym = sims.union(sims.select(F.col("j").alias("i"), F.col("i").alias("j"), "sim"))
w2 = Window.partitionBy("i").orderBy(F.abs("sim").desc())
nbrs = sym.withColumn("r", F.row_number().over(w2)).filter("r <= 50").drop("r")

nbrs.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_neighbors")
nbrs = spark.table("workspace.default.item_neighbors")

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

v = (val.select("user_idx", F.col("item_idx").alias("i"), "rating")
     .join(user_b, "user_idx", "left")
     .join(item_b.withColumnRenamed("item_idx", "i"), "i", "left")
     .fillna({"b_u": 0.0, "b_i": 0.0})
     .withColumn("base", F.lit(mu) + F.col("b_u") + F.col("b_i")))

hist = tr.select("user_idx", F.col("item_idx").alias("j"), "e")   # what the user rated in train

c = v.join(hist, "user_idx").join(nbrs, ["i", "j"])   # neighbors of i that the user rated

# keep the 20 most similar rated neighbors per (user, item)
w3 = Window.partitionBy("user_idx", "i").orderBy(F.abs("sim").desc())
c = c.withColumn("r", F.row_number().over(w3)).filter("r <= 20")

adj = (c.groupBy("user_idx", "i")
       .agg((F.sum(F.col("sim") * F.col("e")) / F.sum(F.abs("sim"))).alias("adj")))

pred = (v.join(adj, ["user_idx", "i"], "left")
        .fillna({"adj": 0.0})                         # no neighbors -> just the baseline
        .withColumn("prediction", F.least(F.lit(5.0), F.greatest(F.lit(1.0),
                                  F.col("base") + F.col("adj")))))

print("item-item CF RMSE:", evaluator.evaluate(pred))
print("rows with neighbors:", adj.count(), "of", v.count())

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.ml.evaluation import RegressionEvaluator

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")

ev = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
reg_values, min_overlap_values = [5, 10, 20, 40], [2, 3, 5]
item_limit = 50

a = tr.select("user_idx", F.col("item_idx").alias("i"), F.col("e").alias("ei"))
b = tr.select("user_idx", F.col("item_idx").alias("j"), F.col("e").alias("ej"))
hist = tr.select("user_idx", F.col("item_idx").alias("j"), "e")

clip = lambda x: F.least(F.lit(5.), F.greatest(F.lit(1.), x))
base_rmse = ev.evaluate(v.withColumn("prediction", clip(F.col("base"))))
print(f"Baseline: {base_rmse:.4f}")

results = []
for min_overlap in min_overlap_values:
    for lam in reg_values:
        sims = (a.join(b, "user_idx").filter("i < j").groupBy("i", "j")
            .agg(F.count("*").alias("n"),
                 F.sum(F.col("ei")*F.col("ej")).alias("dot"),
                 F.sum(F.col("ei")**2).alias("ni"),
                 F.sum(F.col("ej")**2).alias("nj"))
            .filter(F.col("n") >= min_overlap)
            .withColumn("sim", F.col("dot")/F.sqrt(F.col("ni")*F.col("nj"))
                        * F.col("n")/(F.col("n")+lam))
            .select("i", "j", "sim"))

        sym = sims.unionByName(
            sims.select(F.col("j").alias("i"), F.col("i").alias("j"), "sim"))
        wi = Window.partitionBy("i").orderBy(F.abs("sim").desc())
        nbrs = (sym.withColumn("rn", F.row_number().over(wi))
                    .filter(F.col("rn") <= item_limit).drop("rn"))

        c = v.join(hist, "user_idx").join(nbrs, ["i", "j"])
        adj = c.groupBy("user_idx", "i").agg(
            (F.sum(F.col("sim")*F.col("e")) /
             F.sum(F.abs("sim"))).alias("adj"))

        pred = (v.join(adj, ["user_idx", "i"], "left").fillna({"adj": 0.})
                .withColumn("prediction", clip(F.col("base")+F.col("adj"))))

        rmse = ev.evaluate(pred)
        results.append((min_overlap, lam, rmse))
        print(f"min_overlap={min_overlap}, lambda={lam:>2}: "
              f"RMSE={rmse:.4f} (baseline {base_rmse:.4f})")

spark.createDataFrame(results, ["min_overlap", "lambda_sim", "val_rmse"]) \
    .orderBy("val_rmse").show(12, False)

# COMMAND ----------

# MAGIC %md
# MAGIC with posstive similarities:

# COMMAND ----------

# MAGIC %md
# MAGIC item-item CF RMSE: 1.3793280500138203
# MAGIC rows with neighbors: 17865 of 665176
# MAGIC For filter 5
# MAGIC
# MAGIC
# MAGIC item-item CF RMSE: 1.2577003077119824
# MAGIC rows with neighbors: 24914 of 665176
# MAGIC for filter 3
# MAGIC
# MAGIC item-item CF RMSE: 1.2604129917939566
# MAGIC rows with neighbors: 32037 of 665176
# MAGIC for filter 2

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.ml.evaluation import RegressionEvaluator

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")

ev = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
reg_values, min_overlap_values = [1, 3, 5], [5, 10, 15]
item_limit = 50

a = tr.select("user_idx", F.col("item_idx").alias("i"), F.col("e").alias("ei"))
b = tr.select("user_idx", F.col("item_idx").alias("j"), F.col("e").alias("ej"))
hist = tr.select("user_idx", F.col("item_idx").alias("j"), "e")

clip = lambda x: F.least(F.lit(5.), F.greatest(F.lit(1.), x))
base_rmse = ev.evaluate(v.withColumn("prediction", clip(F.col("base"))))
print(f"Baseline: {base_rmse:.4f}")

results = []
for min_overlap in min_overlap_values:
    for lam in reg_values:
        sims = (a.join(b, "user_idx").filter("i < j").groupBy("i", "j")
            .agg(F.count("*").alias("n"),
                 F.sum(F.col("ei")*F.col("ej")).alias("dot"),
                 F.sum(F.col("ei")**2).alias("ni"),
                 F.sum(F.col("ej")**2).alias("nj"))
            .filter(F.col("n") >= min_overlap)
            .withColumn("sim", F.col("dot")/F.sqrt(F.col("ni")*F.col("nj"))
                        * F.col("n")/(F.col("n")+lam))
            .select("i", "j", "sim"))

        sym = sims.unionByName(
            sims.select(F.col("j").alias("i"), F.col("i").alias("j"), "sim"))
        wi = Window.partitionBy("i").orderBy(F.abs("sim").desc())
        nbrs = (sym.withColumn("rn", F.row_number().over(wi))
                    .filter(F.col("rn") <= item_limit).drop("rn"))

        c = v.join(hist, "user_idx").join(nbrs, ["i", "j"])
        adj = c.groupBy("user_idx", "i").agg(
            (F.sum(F.col("sim")*F.col("e")) /
             F.sum(F.abs("sim"))).alias("adj"))

        pred = (v.join(adj, ["user_idx", "i"], "left").fillna({"adj": 0.})
                .withColumn("prediction", clip(F.col("base")+F.col("adj"))))

        rmse = ev.evaluate(pred)
        results.append((min_overlap, lam, rmse))
        print(f"min_overlap={min_overlap}, lambda={lam:>2}: "
              f"RMSE={rmse:.4f} (baseline {base_rmse:.4f})")

spark.createDataFrame(results, ["min_overlap", "lambda_sim", "val_rmse"]) \
    .orderBy("val_rmse").show(12, False)

# COMMAND ----------

from pyspark.sql import functions as F


a = tr.select("user_idx", F.col("item_idx").alias("i"))
b = tr.select("user_idx", F.col("item_idx").alias("j"))

pairs = (a.join(b, "user_idx")
           .filter(F.col("i") < F.col("j"))
           .groupBy("i", "j")
           .agg(F.count("*").alias("n_common")))

# Item pairs with more than 15 common users
strong_pairs = pairs.filter(F.col("n_common") > 15)

print(f"Total item pairs: {pairs.count():,}")
print(f"Pairs with >15 common users: {strong_pairs.count():,}")

# Distribution of common-user counts
pairs.select(
    F.when(F.col("n_common") > 15, ">15")
     .otherwise("<=15").alias("group")
).groupBy("group").count().show()

# Number of distinct items involved in those pairs
items_in_pairs = (
    strong_pairs.select(F.col("i").alias("item_idx"))
    .union(strong_pairs.select(F.col("j").alias("item_idx")))
    .distinct()
    .count()
)
print(f"Distinct items in at least one >15-common-user pair: {items_in_pairs:,}")

# COMMAND ----------

total_train_items = tr.select("item_idx").distinct().count()
print(f"Total distinct items in train (tr): {total_train_items:,}")

# COMMAND ----------

from pyspark.sql import functions as F

# Item pairs based on training history
a = tr.select("user_idx", F.col("item_idx").alias("i"))
b = tr.select("user_idx", F.col("item_idx").alias("j"))

pairs = (a.join(b, "user_idx")
           .filter(F.col("i") < F.col("j"))
           .groupBy("i", "j")
           .agg(F.count("*").alias("n_common")))

# Distinct items in pairs with >15 common users
strong_pairs = pairs.filter(F.col("n_common") > 15)

strong_items = (
    strong_pairs.select(F.col("i").alias("item_idx"))
    .union(strong_pairs.select(F.col("j").alias("item_idx")))
    .distinct()
)

# Distinct items appearing in validation
val_items = val.select("item_idx").distinct()

# Items that occur in both sets
items_in_val = strong_items.join(val_items, "item_idx", "inner").count()

total_strong_items = strong_items.count()
total_val_items = val_items.count()

print(f"Distinct items in train pairs with >15 common users: {total_strong_items:,}")
print(f"Distinct items in validation: {total_val_items:,}")
print(f"Strong-pair items also appearing in validation: {items_in_val:,}")
print(f"Percentage of strong-pair items appearing in validation: "
      f"{100 * items_in_val / total_strong_items:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC Now blend

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")


tr = (train.join(item_b, "item_idx").join(user_b, "user_idx")
      .withColumn("e", F.col("rating") - (F.lit(mu) + F.col("b_i") + F.col("b_u")))
      .select("user_idx", "item_idx", "e", "timestamp"))

w = Window.partitionBy("user_idx").orderBy(F.col("timestamp").desc())
tr = tr.withColumn("rn", F.row_number().over(w)).filter("rn <= 100").drop("rn")

a = tr.select("user_idx", F.col("item_idx").alias("i"), F.col("e").alias("ei"))
b = tr.select("user_idx", F.col("item_idx").alias("j"), F.col("e").alias("ej"))

lam2 = 1
sims = (a.join(b, "user_idx").filter("i < j")        
        .groupBy("i", "j")
        .agg(F.count("*").alias("n"),
             F.sum(F.col("ei") * F.col("ej")).alias("dot"),
             F.sum(F.col("ei")**2).alias("ni"),
             F.sum(F.col("ej")**2).alias("nj"))
        .filter("n >= 15")   #Experiment with 2,3,4                       
        .withColumn("sim", F.col("dot") / F.sqrt(F.col("ni") * F.col("nj"))
                           * F.col("n") / (F.col("n") + lam2))   
        .select("i", "j", "sim"))

# make it symmetric, keep the 50 strongest neighbors per item
sym = sims.union(sims.select(F.col("j").alias("i"), F.col("i").alias("j"), "sim"))
w2 = Window.partitionBy("i").orderBy(F.abs("sim").desc())
nbrs = sym.withColumn("r", F.row_number().over(w2)).filter("r <= 50").drop("r")

nbrs.write.format("delta").mode("overwrite").saveAsTable("workspace.default.item_neighbors")
nbrs = spark.table("workspace.default.item_neighbors")

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

train = spark.table("workspace.default.amazon_train")
val   = spark.table("workspace.default.amazon_val")
item_b = spark.table("workspace.default.final_item_b")
user_b = spark.table("workspace.default.final_user_b")

def add_base(d):
    return (d.join(item_b, "item_idx", "left").join(user_b, "user_idx", "left")
             .fillna({"b_i": 0.0, "b_u": 0.0})
             .withColumn("base", F.lit(mu) + F.col("b_i") + F.col("b_u")))

train_b = add_base(train).withColumn("resid", F.col("rating") - F.col("base"))
val_b   = add_base(val)

als_best = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="resid",
               rank=64, regParam=0.3, maxIter=15,
               coldStartStrategy="nan", seed=42).fit(train_b)

def als_adj(d):
    return (als_best.transform(d.select("user_idx", "item_idx"))
            .withColumn("als_adj", F.when(F.isnan("prediction"), 0.0).otherwise(F.col("prediction")))
            .select("user_idx", "item_idx", "als_adj"))

def cf_adj(d):
    v = d.select("user_idx", F.col("item_idx").alias("i"))
    c = v.join(hist, "user_idx").join(nbrs, ["i", "j"])
    c = c.withColumn("r", F.row_number().over(w3)).filter("r <= 20")
    return (c.groupBy("user_idx", "i")
              .agg((F.sum(F.col("sim") * F.col("e")) / F.sum(F.abs("sim"))).alias("cf_adj"))
              .withColumnRenamed("i", "item_idx"))
def build(d):
    return (add_base(d)   # adds base = mu + item bias + user bias
            .join(als_adj(d), ["user_idx", "item_idx"], "left")
            .join(cf_adj(d), ["user_idx", "item_idx"], "left")
            .fillna({"als_adj": 0.0, "cf_adj": 0.0}))

build(val).write.format("delta").mode("overwrite").saveAsTable("workspace.default.blend_val")
build(test).write.format("delta").mode("overwrite").saveAsTable("workspace.default.blend_test")

val_f, test_f = spark.table("workspace.default.blend_val"), spark.table("workspace.default.blend_test")
print(val_f.count(), val.count())     # must be equal: no duplicated or lost rows

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

asm = VectorAssembler(inputCols=["base", "als_adj", "cf_adj"], outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="rating",
                      regParam=0.01, elasticNetParam=0.0)   # pure ridge
blend = lr.fit(asm.transform(val_f))
print(blend.coefficients, blend.intercept)

clip = lambda c: F.least(F.lit(5.0), F.greatest(F.lit(1.0), c))
rm = lambda d, col: RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol=col).evaluate(d)

t = (blend.transform(asm.transform(test_f))
     .withColumn("blend", clip(F.col("prediction")))
     .withColumn("p_base", clip(F.col("base")))
     .withColumn("p_als", clip(F.col("base") + F.col("als_adj")))
     .withColumn("p_cf",  clip(F.col("base") + F.col("cf_adj"))))

for name in ["p_base", "p_als", "p_cf", "blend"]:
    print(name, round(rm(t, name), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC [0.9437348255551802,-0.0020601979147590044,0.01580216764803444] 0.20558378091576798
# MAGIC p_base 1.157
# MAGIC p_als 1.1604
# MAGIC p_cf 1.161
# MAGIC blend 1.1585

# COMMAND ----------

# MAGIC %md
# MAGIC [0.943641925946155,-0.0034900570625601667,0.02604168367688302] 0.20596048516418497
# MAGIC p_base 1.157
# MAGIC p_als 1.1604
# MAGIC p_cf 1.164
# MAGIC blend 1.1585
# MAGIC for neighbours=2

# COMMAND ----------

tr_items = train.select("item_idx").distinct()
unseen_item_rows = val.join(tr_items, "item_idx", "left_anti").count()
print(unseen_item_rows, "of", val.count())

# COMMAND ----------

# MAGIC %md
# MAGIC olculsion: the ALS does not contribute with misleading noise and will be removed for futur

# COMMAND ----------

for name, d in [("val", val_f), ("test", test_f)]:
    c = d.filter(F.col("cf_adj") != 0)
    alpha = c.agg((F.sum((F.col("rating") - F.col("base")) * F.col("cf_adj"))
                   / F.sum(F.col("cf_adj")**2))).first()[0]
    c = (c.withColumn("p_base", clip(F.col("base")))
          .withColumn("p_cf_a", clip(F.col("base") + alpha * F.col("cf_adj"))))
    print(name, "rows:", c.count(), "alpha:", round(alpha, 3),
          "base:", round(rm(c, "p_base"), 4), "CF:", round(rm(c, "p_cf_a"), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC for neighbours=2 val rows: 32037 alpha: 0.025 base: 1.2573 CF: 1.257
# MAGIC test rows: 15850 alpha: 0.043 base: 1.2821 CF: 1.2814

# COMMAND ----------

from pyspark.sql import functions as F

tr_users = train.select("user_idx").distinct().withColumn("u_seen", F.lit(1))
tr_items = train.select("item_idx").distinct().withColumn("i_seen", F.lit(1))

def groups(d, name):
    g = (d.join(tr_users, "user_idx", "left").join(tr_items, "item_idx", "left")
          .withColumn("user", F.when(F.col("u_seen") == 1, "seen user").otherwise("new user"))
          .withColumn("item", F.when(F.col("i_seen") == 1, "seen item").otherwise("new item")))
    total = g.count()
    out = (g.groupBy("user", "item").count()
            .withColumn("share_%", F.round(F.col("count") / total * 100, 1))
            .withColumn("split", F.lit(name)))
    return out

res = groups(val, "val").unionByName(groups(test, "test"))
res.orderBy("split", "user", "item").show()

# the specific number you asked for: seen user, new item
res.filter("user = 'seen user' AND item = 'new item'").show()

# COMMAND ----------

# MAGIC %md
# MAGIC