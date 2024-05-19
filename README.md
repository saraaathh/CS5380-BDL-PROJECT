# *Fashion-MNIST*

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of *60,000* examples and a test set of *10,000* examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. It's intended that Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

This project aims to build a machine learning model on Fashion-MNIST using Apache Spark for data processing, Apache Airflow for pipeline orchestration, MLflow for experiment tracking, and a REST API for model serving. The project also includes Prometheus for metrics instrumentation and Grafana for visualization.

## *Contents*
- Apache Spark
- Apache Airflow
- MLflow
- Prometheus
- Grafana
## *Data Pipeline*
Apache Airflow is used to orchestrate the data pipeline. The pipeline consists of following tasks:
1. *Data Loading* -
2. *Data Preprocessing* -
3. *Data Saving* -

## *Model Building*
MLFLOW
## *Model Tracking*
MLFLOW

## *REST API*
USING FATSAPI
## *Metrics and Monitoring*
Prometheus is used for metrics instrumentation, and the configuration is defined in the prometheus/filename.yml file. The REST API functions are instrumented to capture relevant metrics.

Grafana is used for visualizing the captured metrics. The grafana/directory name directory contains the Grafana dashboard configuration files, and the grafana/directory name directory contains the data source configuration files.

## *Deployment*
The project can be deployed using Docker. The Dockerfile defines the Docker image for the REST API, and the docker-compose.yml file defines the services for the REST API, Prometheus, and Grafana.

Following steps need to be followed, to deploy the project:

1. Build the docker image - docker-compose build
2. Start the services - docker-compose up -d
