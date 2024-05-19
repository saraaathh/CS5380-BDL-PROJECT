from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def preprocess_data(train_spark_df, test_spark_df):
    # Check for missing values and drop rows with any missing values
    train_spark_df = train_spark_df.na.drop()
    test_spark_df = test_spark_df.na.drop()
    
    # Normalize the pixel values (0-255 to 0-1)
    for col_name in train_spark_df.columns:
        if col_name != 'label':
            train_spark_df = train_spark_df.withColumn(col_name, col(col_name) / 255.0)
            test_spark_df = test_spark_df.withColumn(col_name, col(col_name) / 255.0)
    
    return train_spark_df, test_spark_df

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Fashion MNIST Preprocessing").getOrCreate()
    
    # Load raw data
    train_spark_df = spark.read.parquet('train_raw_data.parquet')
    test_spark_df = spark.read.parquet('test_raw_data.parquet')
    
    # Preprocess data
    train_spark_df, test_spark_df = preprocess_data(train_spark_df, test_spark_df)
    
    # Save the preprocessed data
    train_spark_df.write.mode('overwrite').parquet('train_preprocessed_data.parquet')
    test_spark_df.write.mode('overwrite').parquet('test_preprocessed_data.parquet')
