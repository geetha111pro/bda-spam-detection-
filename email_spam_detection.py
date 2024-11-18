from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col
import os

# Update environment variables for Render deployment
os.environ["PYSPARK_PYTHON"] = "/opt/render/.python/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/opt/render/.python/bin/python"

# Initialize Spark session
spark = SparkSession.builder \
    .appName("EmailSpamDetection") \
    .getOrCreate()

# Load data
data = spark.read.csv('spam.csv', header=True, inferSchema=True)

# Check if 'v1' and 'v2' columns exist
if 'v1' not in data.columns or 'v2' not in data.columns:
    raise ValueError("Input data does not contain expected columns 'v1' and 'v2'.")

# Rename columns
data = data.withColumnRenamed("v1", "label").withColumnRenamed("v2", "text")

# Convert labels
data = data.replace({"ham": "0", "spam": "1"}, subset=["label"])
data = data.withColumn("label", col("label").cast("int"))

# Filter out rows with null labels or text
data = data.filter(col("label").isNotNull() & col("text").isNotNull())

# Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
data = tokenizer.transform(data)

# Stopwords Removal
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
data = stopwords_remover.transform(data)

# Term Frequency
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
data = hashing_tf.transform(data)

# Inverse Document Frequency (IDF)
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(data)
data = idf_model.transform(data)

# Train-Test Split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Models
lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train)

# Prediction Function
def predict_spam_or_ham(input_text):
    # Convert input text to DataFrame
    input_data = spark.createDataFrame([(input_text,)], ["text"])

    # Apply transformations
    input_data = tokenizer.transform(input_data)
    input_data = stopwords_remover.transform(input_data)
    input_data = hashing_tf.transform(input_data)
    input_data = idf_model.transform(input_data)

    # Logistic Regression Prediction
    lr_prediction = lr_model.transform(input_data)
    lr_pred = lr_prediction.select("prediction").collect()[0][0]
    
    # Return the result based on the Logistic Regression model prediction
    return "ham" if lr_pred == 0 else "spam"
