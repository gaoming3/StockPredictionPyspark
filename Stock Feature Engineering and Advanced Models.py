# Databricks notebook source
from pyspark import keyword_only  ## < 2.0 -> pyspark.ml.util.keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

from pyspark.sql import SQLContext, Window
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.evaluation import RegressionEvaluator
from  pyspark.sql.functions import abs, sqrt
import pandas as pd

#CUSTOM TRANSFORMERS
class StockFeatureCreator(Transformer):
    
    @keyword_only
    def __init__(self, lags, pred_window, ma_windows, tickers):
        self._paramMap = {}

        #: internal param map for default values
        self._defaultParamMap = {}

        #: value returned by :py:func:`params`
        self._params = None

        # Copy the params from the class to the object
        self._copy_params()
        
        self.lags = lags
        self.pred_window = pred_window
        self.ma_windows = ma_windows
        self.inputCols = []
        self.tickers = tickers
        self.outputCols = []
        
    def getOutputCols(self):
      
        return self.outputCols
    
    def build(self):
        
        #GENERATE ALL COLUMN NAMES
        for tick in self.tickers:
          self.inputCols.append("Open_"+tick)
          self.inputCols.append("High_"+tick)
          self.inputCols.append("Low_"+tick)
          self.inputCols.append("Volume_"+tick)
          self.inputCols.append("Close_"+tick)
          self.inputCols.append("Volume_"+tick)
        
        #Lags
        for lag_num in range(1,self.lags+1):
          for feature in self.inputCols:
            self.outputCols.append("{}_Lag_{}".format(feature,lag_num))
        
        #MA windows
        for ma_length in self.ma_windows:
          for feature in self.inputCols:  
            self.outputCols.append("{}_ma_{}".format(feature,ma_length))
            self.outputCols.append("{}_var_{}".format(feature,ma_length))
            self.outputCols.append("{}_Z_{}".format(feature,ma_length))
        
        for lag_num in range(1,self.lags+1):
          for tick in self.tickers: 
            self.outputCols.append("Open_Close_{}_Ratio_Lag_{}".format(tick,lag_num))
        
        for lag_num in range(1,int((self.lags)/3)):
          for feature in self.inputCols:
            self.outputCols.append("{}_Diff_{}".format(feature,lag_num))
            self.outputCols.append("{}_Diff_Percent_{}".format(feature,lag_num))
            self.outputCols.append("{}_Diff_{}_Sign".format(feature,lag_num))
            self.outputCols.append("{}_Rolling_Sign_{}".format(feature,lag_num))
        
        #self.outputCols = [s + "_imputed" for s in self.outputCols]
        
    def _transform(self, dataset):
        
        #LOG VOLUME
        for tick in self.tickers:
          dataset = dataset.withColumn("Volume_{}_float".format(tick), log(col("Volume_{}".format(tick)).cast(DoubleType()))) #log because large value
          dataset = dataset.drop("Volume_{}".format(tick))
          dataset = dataset.withColumnRenamed("Volume_{}_float".format(tick),"Volume_{}".format(tick))
        
        #GENERATE TIME COLUMNS
        dataset = dataset.withColumn("Day_Of_Week",dayofweek("Date"))
        dataset = dataset.withColumn("Month",month("Date"))
        dataset = dataset.withColumn("Quarter",quarter(col("Date")))
        dataset = dataset.withColumn("Week_Of_Year",weekofyear(col("Date")))
        #dataset = dataset.withColumn("Day_Of_Year",dayofyear(col("Date")))
        dataset = dataset.withColumn("Day_Of_Month",dayofmonth(col("Date")))
        #dataset = dataset.withColumn("Year",year(col("Date")))
        
        time_cols = ["Date",
                     "Day_Of_Week",
                     "Month",
                     "Quarter",
                     "Week_Of_Year",
                     "Day_Of_Month"]
        
        #GENERATE LAG COLUMNS
        for lag_num in range(1,self.lags+1):
          for feature in self.inputCols:
            dataset = dataset.withColumn("{}_Lag_{}".format(feature,lag_num),
                                          lag(feature,lag_num,None).over(Window.orderBy("Date")))
          
        #SIMPLE MOVING AVERAGES, MOVING VARIANCE AND Z SCORE
        for ma_length in self.ma_windows:
          for feature in self.inputCols:
            dataset = dataset.withColumn("{}_ma_{}".format(feature,ma_length), avg(col(feature)).over(Window.rowsBetween(-ma_length,-1)))
            dataset = dataset.withColumn("{}_var_{}".format(feature,ma_length), variance(col(feature)).over(Window.rowsBetween(-ma_length,-1)))
            dataset = dataset.withColumn("{}_Z_{}".format(feature,ma_length), (col("{}_Lag_1".format(feature)) - col("{}_ma_{}".format(feature,ma_length)))/(col("{}_var_{}".format(feature,ma_length))))
          
        #OPEN/CLOSE RATIO
        for lag_num in range(1,self.lags+1):
          for tick in self.tickers:
            dataset = dataset.withColumn("Open_Close_{}_Ratio_Lag_{}".format(tick,lag_num),col("Open_{}_Lag_{}".format(tick,lag_num))/col("Close_{}_Lag_{}".format(tick,lag_num)))
        
        #DIFFERENCING, PERCENT CHANGE, SIGN AND ROLLING SUM 
        for lag_num in range(1,int((self.lags)/3)):
          for feature in self.inputCols:
            dataset = dataset.withColumn("{}_Diff_{}".format(feature,lag_num), when(isnull(col("{}_Lag_{}".format(feature,lag_num)) - col("{}_Lag_{}".format(feature,lag_num+1))), 0).otherwise(col("{}_Lag_{}".format(feature,lag_num)) - col("{}_Lag_{}".format(feature,lag_num+1))))
            dataset = dataset.withColumn("{}_Diff_Percent_{}".format(feature,lag_num), when(isnull((col("{}_Lag_{}".format(feature,lag_num)) - col("{}_Lag_{}".format(feature,lag_num+1)))/col("{}_Lag_{}".format(feature,lag_num+1))), 0).otherwise((col("{}_Lag_{}".format(feature,lag_num)) - col("{}_Lag_{}".format(feature,lag_num+1)))/col("{}_Lag_{}".format(feature,lag_num+1))))
            dataset = dataset.withColumn("{}_Diff_{}_Sign".format(feature,lag_num), when(col("{}_Diff_{}".format(feature,lag_num)) > 0, 1.0).otherwise(-1.0))
            dataset = dataset.withColumn("{}_Rolling_Sign_{}".format(feature,lag_num), sum(col("{}_Diff_{}_Sign".format(feature,lag_num))).over(Window.rowsBetween(-(lag_num+1),-1)))
        
        #IMPUTE VALUES
        """imputer = Imputer(inputCols=[column for column in dataset.columns if column not in time_cols], 
                           outputCols=["{}_imputed".format(c) for c in [column for column in dataset.columns if column not in time_cols]],
                           strategy = "median")
        dataset = imputer.fit(dataset).transform(dataset)
        dataset = dataset.dropna()"""
        
        #Drop Columns
        drop_list_fil = [col for col in self.inputCols if col != "Close_aapl"]
        drop_list = [col for col in drop_list_fil]
        
        dataset = dataset.select([col for col in dataset.columns if col not in drop_list])

        #PREDICTION WINDOW
        dataset = dataset.withColumn("Close_aapl_Window",lead("Close_aapl",self.pred_window-1,None).over(Window.orderBy("Date")))
        dataset = dataset.drop("Close_aapl")
        dataset = dataset.withColumnRenamed("Close_aapl_Window","Close_aapl")
        dataset  = dataset.withColumnRenamed("Close_aapl","label")

        #DROP NULL CREATED BY 
        dataset = dataset.dropna()
        
        #REMOVE NON IMPUTED COLUMNS
        #dataset  = dataset.select([col for col in dataset.columns if ("_imputed" in col) or (col in time_cols)])
        
        #self.outputCols = dataset.columns
        
        return dataset   


# COMMAND ----------

#FUNCTIONS
def evaluateModel(model,data):
  pred_results = model.transform(data)

  pred_results = pred_results.withColumn("squared_error",pow((col("label") - col("prediction")),2))
  pred_results = pred_results.withColumn("s_abs_percentage_error",(abs(col("prediction") - col("label"))/((col("label") + col("prediction"))/2))*100)

  total_rmse = pred_results.select(sqrt(mean(col("squared_error")))).collect()
  total_smape = pred_results.select(mean(col("s_abs_percentage_error"))).collect()

  #print("Train RMSE: ", model.bestModel.summary.rootMeanSquaredError)
  print("Test RMSE: ", total_rmse)
  print("Test sMAPE: ", total_smape)
  
  return pred_results

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))


# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, StandardScaler, OneHotEncoderEstimator, PCA, VectorSlicer, ChiSqSelector
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd

time_cols = ["Date",
             "Day_Of_Week",
             "Month",
             "Quarter",
             "Week_Of_Year",
             "Day_Of_Month"]

#READ DATA
sqlContext = SQLContext(sc)
df = sqlContext.sql("""
                    SELECT aapl.Date, aapl.Open as Open_aapl, aapl.High as High_aapl, aapl.Low as Low_aapl, aapl.Close as Close_aapl, aapl.Volume as Volume_aapl,
                    sp500_csv.Open as Open_sp500, sp500_csv.High as High_sp500, sp500_csv.Low as Low_sp500, sp500_csv.Close as Close_sp500, sp500_csv.Volume as Volume_sp500
                    FROM aapl
                    JOIN sp500_csv on aapl.Date = sp500_csv.Date
                    WHERE YEAR(aapl.Date) >= 2009 and YEAR(aapl.Date) <= 2015
                    """)

#msft.Open as Open_msft, msft.High as High_msft, msft.Low as Low_msft, msft.Close as Close_msft, msft.Volume as Volume_msft
#JOIN msft on aapl.Date = msft.Date


#BUILD FEATURES
#Pred windows 1, 5, 10, 20, 60
stockcreator = StockFeatureCreator(lags = 10, pred_window = 1, ma_windows = [3,5,10,20,50,80,100], tickers = ["aapl","sp500"])
stockcreator.build()

#ONE HOT CATEGORICAL DATE FEATURES
inputs = [s for s in time_cols if s not in ["Date"]]
encoder = OneHotEncoderEstimator(inputCols=inputs, outputCols=[s + "_Vec" for s in time_cols if s not in ["Date"]])

#VECTOR ASSEMBLER
features = stockcreator.getOutputCols() + encoder.getOutputCols() #getOutoutCols returning empty list
features = [col for col in features if (col != "label") or (col not in time_cols)]
featureassembler = VectorAssembler(inputCols=features,outputCol="features")

#SPLIT
finalized_data = df.withColumn("rank",percent_rank().over(Window.partitionBy().orderBy("Date")))
train_data = finalized_data.where("rank <= .9").drop("rank")
test_data = finalized_data.where("rank > .9").drop("rank")

#FEATURE SELECTION
selector = ChiSqSelector(numTopFeatures=300, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="label")

'''num_features = 50

fs_estimator = RandomForestRegressor(labelCol="label", featuresCol="all_features")
pipe1 = Pipeline(stages=[stockcreator, encoder, featureassembler, fs_estimator])
model = pipe1.fit(train_data)
df2 = model.transform(test_data)

varlist = ExtractFeatureImp(model.stages[-1].featureImportances, df2, "all_features")
varidx = [x for x in varlist['idx'][0:num_features]]
slicer = VectorSlicer(inputCol="all_features", outputCol="features", indices=varidx)
df2.unpersist()'''

#FIT MODEL
estimator = GBTRegressor(labelCol="label", featuresCol="selectedFeatures") #can be changed to RF, or RLR - make sure to change param grid for each model

paramGrid = ParamGridBuilder().addGrid(estimator.maxDepth, [2,3,5]).addGrid(estimator.subsamplingRate, [0.8,0.7]).addGrid(estimator.maxIter, [20,30,50]).addGrid(selector.numTopFeatures, [200,300,400]).build()

pipe2 = Pipeline(stages=[stockcreator, encoder, featureassembler, selector, estimator])

crossval = CrossValidator(estimator=pipe2,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator().setMetricName("rmse"),
                          numFolds=3)

cvModel = crossval.fit(train_data)

#PRINT RESULTS
params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]

pd.DataFrame.from_dict([
    {cvModel.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, cvModel.avgMetrics)
])


# COMMAND ----------

# MAGIC %md Example Runs (Not Including all tests for all models)

# COMMAND ----------

# MAGIC %md 1 Day GBT 300 Features

# COMMAND ----------

results = evaluateModel(cvModel.bestModel,train_data)
ExtractFeatureImp(cvModel.bestModel.stages[-1].featureImportances, results, "selectedFeatures")

# COMMAND ----------

# MAGIC %md 4 Month GBT 300 Features

# COMMAND ----------

results = evaluateModel(cvModel.bestModel,train_data)
ExtractFeatureImp(cvModel.bestModel.stages[-1].featureImportances, results, "selectedFeatures")

# COMMAND ----------

results.select("features","label","prediction").show()

# COMMAND ----------


