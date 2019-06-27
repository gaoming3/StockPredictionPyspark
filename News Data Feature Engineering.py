from pyspark.sql import SQLContext, Window
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.evaluation import RegressionEvaluator
from  pyspark.sql.functions import abs, sqrt
from pyspark.sql.functions import concat, col, lit
from pyspark.ml.feature import StopWordsRemover, Tokenizer, RegexTokenizer, HashingTF, IDF

#TFIDF BOW FEATURES

df_news = sqlContext.sql("SELECT Date, Top1,Top2,Top25 FROM combined_news_djia_csv")

num_word_features = 2000

#news data only goes to july 2016
df_news = sqlContext.sql("SELECT * FROM combined_news_djia_csv")
df_news = df_news.select("Date",concat(col("Top1"), lit(" "), col("Top2"), lit(" "), col("Top3"), lit(" "), col("Top4"), lit(" "), col("Top5"), lit(" "), col("Top6"), lit(" "), col("Top7"), lit(" "), col("Top8"), lit(" "), col("Top9"), lit(" "), col("Top10"), lit(" "), col("Top11"), lit(" "), col("Top12"), lit(" "), col("Top13"), lit(" "), col("Top14"), lit(" "), col("Top15"), lit(" "), col("Top16"), lit(" "), col("Top17"), lit(" "), col("Top18"), lit(" "), col("Top19"), lit(" "), col("Top20"), lit(" "), col("Top21"), lit(" "), col("Top22"), lit(" "), col("Top23"), lit(" "), col("Top24"), lit(" "), col("Top25")).alias("all_text_dirty"))

df_news = df_news.withColumn("all_text_1",regexp_replace(col("all_text_dirty"), "['\"]", ""))
df_news = df_news.withColumn("all_text",expr("substring(all_text_1, 2, length(all_text_1)+1)"))


df_news = df_news.dropna()

tokenizer = Tokenizer(inputCol="all_text", outputCol="words")
wordsData = tokenizer.transform(df_news)

remover = StopWordsRemover(inputCol="words", outputCol="wordsFil")
wordsDataFil = remover.transform(wordsData)

hashingTF = HashingTF(inputCol="wordsFil", outputCol="rawFeatures", numFeatures=num_word_features)
featurizedData = hashingTF.transform(wordsDataFil)
# alternatively, CountVectorizer can also be used to get term frequency vectors

idf = IDF(inputCol="rawFeatures", outputCol="news_features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

#df_news = rescaledData.select("Date","news_features")

"""imputer = Imputer(
    inputCols=[column for column in df_news.columns if column not in ["Date"]], 
    outputCols=["{}_cleaned".format(c) for c in [column for column in df_news.columns if column not in ["Date"]]],
    strategy = "median"
)

df_news = imputer.fit(df_news).transform(df_news)
df_news = df_news.dropna()"""

#df_news  = df_news.select([col for col in df_news.columns if ("_cleaned" in col) or (col in time_cols)])

df_news = rescaledData.select("news_features")

#NEWS SENTIMENT DATA (FROM KAGGLE AND VADER)
lags = 10

df_news_sent = sqlContext.sql("SELECT Date, Subjectivity, Objectivity, Positive, Neutral, Negative FROM news_sentiment")

#GENERATE LAGS (News might take time to take effect)
for lag_num in range(1,lags+1): 
    df_news_sent = df_news_sent.withColumn("Subjectivity_Lag_{}".format(lag_num),avg(col("Subjectivity")).over(Window.rowsBetween(-lag_num,-lag_num)))
    df_news_sent = df_news_sent.withColumn("Objectivity_Lag_{}".format(lag_num),avg(col("Objectivity")).over(Window.rowsBetween(-lag_num,-lag_num)))
    df_news_sent = df_news_sent.withColumn("Positive_Lag_{}".format(lag_num),avg(col("Positive")).over(Window.rowsBetween(-lag_num,-lag_num)))
    df_news_sent = df_news_sent.withColumn("Neutral_Lag_{}".format(lag_num),avg(col("Neutral")).over(Window.rowsBetween(-lag_num,-lag_num)))
    df_news_sent = df_news_sent.withColumn("Negative_Lag_{}".format(lag_num),avg(col("Negative")).over(Window.rowsBetween(-lag_num,-lag_num)))

drop_list = ["Subjectivity","Objectivity","Positive","Neutral","Negative"]
df_news_sent  = df_news_sent.select([column for column in df_news_sent.columns if column not in drop_list])

"""imputer = Imputer(
    inputCols=[column for column in df_news_sent.columns if column not in ["Date"]], 
    outputCols=["{}_clean".format(c) for c in [column for column in df_news_sent.columns if column not in ["Date"]]],
    strategy = "median"
)
df_news_sent = imputer.fit(df_news_sent).transform(df_news_sent)
"""
#df_news_sent = df_news_sent.select([col for col in df_news_sent.columns if ("_clean" in col) or (col in ["Date"])])

df_news_sent = df_news_sent.dropna()
df_news_sent.show(5)