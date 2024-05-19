import os
import requests
from pyspark.sql import SparkSession

def init_spark():
    # Initialize Spark session
    spark = SparkSession.builder.appName("Fashion MNIST Preprocessing").getOrCreate()
    return spark

def download_file(url, local_path):
    response = requests.get(url)
    if response.headers['Content-Type'] == 'text/html':
        raise ValueError("Downloaded content is not CSV, possibly an HTML error page.")
    with open(local_path, 'wb') as file:
        file.write(response.content)

def check_file_content(local_path):
    with open(local_path, 'r') as file:
        for _ in range(5):
            print(file.readline().strip())

def load_data(spark, train_url, test_url, train_local_path, test_local_path):
    # Download the CSV files
    if not os.path.exists(train_local_path):
        print(f"Downloading {train_url} to {train_local_path}")
        download_file(train_url, train_local_path)
    
    if not os.path.exists(test_local_path):
        print(f"Downloading {test_url} to {test_local_path}")
        download_file(test_url, test_local_path)
    
    # Check file content
    print("Checking content of downloaded files:")
    check_file_content(train_local_path)
    check_file_content(test_local_path)
    
    # Read the CSV files
    train_spark_df = spark.read.csv(train_local_path, header=True, inferSchema=True)
    test_spark_df = spark.read.csv(test_local_path, header=True, inferSchema=True)
    
    # Print schema to verify column names
    train_spark_df.printSchema()
    test_spark_df.printSchema()
    
    return train_spark_df, test_spark_df

if __name__ == "__main__":
    train_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion-mnist_train.csv"
    test_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion-mnist_test.csv"
    train_local_path = "fashion-mnist_train.csv"
    test_local_path = "fashion-mnist_test.csv"
    
    spark = init_spark()
    train_spark_df, test_spark_df = load_data(spark, train_url, test_url, train_local_path, test_local_path)
    
    # Save the loaded dataframes for the next steps
    train_spark_df.write.mode('overwrite').parquet('train_raw_data.parquet')
    test_spark_df.write.mode('overwrite').parquet('test_raw_data.parquet')
