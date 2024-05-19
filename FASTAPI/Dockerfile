# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install any dependencies specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define a volume for Grafana storage
VOLUME /path/to/grafana/storage

# Command to run the FastAPI application
CMD ["uvicorn", "restapi_prometheus:app", "--host", "0.0.0.0", "--port", "8000"]


