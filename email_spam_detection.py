from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, PCA
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col
import os


python_path = "C:\\Users\\geeth\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

# Initialize Spark session with Python executable configuration
spark = SparkSession.builder \
    .appName("EmailSpamDetection") \
    .config("spark.executorEnv.PYSPARK_PYTHON", python_path) \
    .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", python_path) \
    .config("spark.driverEnv.PYSPARK_PYTHON", python_path) \
    .config("spark.pyspark.python", python_path) \
    .getOrCreate()
data = spark.read.csv(r'C:\Users\geeth\OneDrive\Desktop\Demo\spam.csv', header=True, inferSchema=True)

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

# Verify tokenization
if "words" not in data.columns:
    raise ValueError("Tokenization failed: 'words' column missing.")

# Stopwords Removal
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
data = stopwords_remover.transform(data)

# Verify stopwords removal
if "filtered_words" not in data.columns:
    raise ValueError("StopWordsRemover failed: 'filtered_words' column missing.")

# Term Frequency
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
data = hashing_tf.transform(data)

# Verify term frequency transformation
if "raw_features" not in data.columns:
    raise ValueError("HashingTF failed: 'raw_features' column missing.")

# Inverse Document Frequency (IDF)
idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(data)
data = idf_model.transform(data)

# Verify IDF transformation
if "features" not in data.columns:
    raise ValueError("IDF transformation failed: 'features' column missing.")

# Train-Test Split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Models
lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train)

rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=10)
rf_model = rf.fit(train)

kmeans = KMeans(k=2, featuresCol="features", predictionCol="prediction")
kmeans_model = kmeans.fit(data)

# PCA for dimensionality reduction (optional)
pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(data)
data_pca = pca_model.transform(data)

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
