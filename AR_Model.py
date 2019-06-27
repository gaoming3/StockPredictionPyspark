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

seed = 26

#Parameters
lags = 80
pred_window = 80

total_mape = []
total_rmse = []

#read in data
sqlContext = SQLContext(sc)
df = sqlContext.sql("SELECT aapl.date, aapl.close AS close1 FROM aapl WHERE YEAR(aapl.Date) >= 2009 and YEAR(aapl.Date) <= 2015")

lag_col_names = ["close1"]

#generate lag columns
for lag_num in range(1,lags+1):
    lagtest = lag("close1",lag_num,0).over(Window.partitionBy().orderBy("date"))
    df = df.withColumn("lag{}".format(lag_num),lagtest)
    lag_col_names.append("lag{}".format(lag_num))

    
#remove null
#df = df.dropna()

#prepare for model
lag_col_names.pop(0) #remove close from name vector

featureassembler = VectorAssembler(inputCols=lag_col_names,outputCol="features")
output = featureassembler.transform(df)
finalized_data = output.select("date","features","close1")

#PREDICTION WINDOW
finalized_data = finalized_data.withColumn("Close_aapl_Window",lead("close1",pred_window-1,None).over(Window.orderBy("date")))
finalized_data = finalized_data.drop("close1")
finalized_data = finalized_data.withColumnRenamed("Close_aapl_Window","close1")
finalized_data = finalized_data.withColumnRenamed("close1","label") 
finalized_data = finalized_data.dropna()


#split into test/train maintaining integrity of time series data
#train_data,test_data = finalized_data.randomSplit([0.7,0.3],seed)
finalized_data = finalized_data.withColumn("rank",percent_rank().over(Window.partitionBy().orderBy("date")))
train_data = finalized_data.where("rank <= .9").drop("rank")
test_data = finalized_data.where("rank > .9").drop("rank")

#fit regression model
regressor = LinearRegression(featuresCol="features",labelCol="label")

paramGrid = ParamGridBuilder().addGrid(regressor.regParam, [0,0.01])\
                              .build()

crossval = CrossValidator(estimator=regressor,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator().setMetricName("rmse"),
                          numFolds=3)

cvModel = crossval.fit(train_data)

#PRINT RESULTS
params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]

cv_results = pd.DataFrame.from_dict([
    {cvModel.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, cvModel.avgMetrics)
])

test_results = evaluateModel(cvModel,test_data)
print(cv_results)
ExtractFeatureImp(cvModel.bestModel.coefficients, test_results, "features").head(100)