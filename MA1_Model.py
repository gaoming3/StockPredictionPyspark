from pyspark.sql import SQLContext, Window
from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import abs, sqrt
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, StandardScaler, OneHotEncoderEstimator, PCA, VectorSlicer
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd

#Parameters
ma_order = 1

#Metrics
total_mape = []
total_rmse = []
total_smape = []

#read in data
sqlContext = SQLContext(sc)
df = sqlContext.sql("SELECT * FROM aapl WHERE YEAR(aapl.date) >= 2009 and YEAR(aapl.date) <= 2015")

#forecast using ma
df = df.withColumn("close_ma", avg(col("close")).over(Window.rowsBetween(-ma_order,-1)))


for window in [1,5,10,20,80]:
  df = df.withColumn("Close_Actual_Window_{}".format(window),lead("close",window-1,None).over(Window.orderBy("Date")))

for window in [1,5,10,20,80]:
  df = df.withColumn("squared_error_Window_{}".format(window), pow((col("Close_Actual_Window_{}".format(window)) - col("close_ma")),2))
  df = df.withColumn("s_abs_percentage_error_Window_{}".format(window),(abs(col("close_ma") -col("Close_Actual_Window_{}".format(window)))/((col("Close_Actual_Window_{}".format(window)) + col("close_ma"))/2))*100)

df.show()

df = df.withColumn("rank",percent_rank().over(Window.partitionBy().orderBy("date")))
train_data = df.where("rank <= .9").drop("rank")
test_data = df.where("rank > .9").drop("rank")

for window in [1,5,10,20,80]:
  total_rmse.append(test_data.select(sqrt(mean(col("squared_error_Window_{}".format(window))))).collect())
  total_smape.append(test_data.select(mean(col("s_abs_percentage_error_Window_{}".format(window)))).collect())

print(total_mape)
print(total_rmse)
print(total_smape)