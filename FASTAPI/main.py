from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
import time
import io 
api_usage_counter = Counter('api_usage_counter', 'API usage counter', ['client_ip'])
api_processing_time_gauge = Gauge('api_processing_time_gauge', 'API processing time gauge', ['client_ip', 'input_length', 'processing_time_per_char'])

# Define a function to preprocess the image and extract pixel values
def preprocess_image(file):
    image = Image.open(io.BytesIO(file)).convert("L")  # Convert the image to grayscale
    resized_image = image.resize((28, 28))   # Resize the image to 28x28
    pixel_values = np.array(resized_image) / 255.0  # Convert pixel values to range [0, 1]
    pixel_values = pixel_values.reshape(1, -1)  # Reshape to match the input shape of the model
    return pixel_values

# Load the pre-trained model
model = load_model("Fashionmnist_best_model.h5")
class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Initialize FastAPI app
app = FastAPI()

# Middleware to collect request metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        body = await request.body()
        request.state.body = body  # Store the body in request state

        response = await call_next(request)
        
        process_time = time.time() - start_time
        input_length = len(body)
        processing_time_per_char = (process_time / input_length) * 1e6 if input_length > 0 else 0
        client_ip = request.client.host

        # Update Prometheus metrics
        api_usage_counter.labels(client_ip=client_ip).inc()
        api_processing_time_gauge.labels(client_ip=client_ip, input_length=input_length, processing_time_per_char=processing_time_per_char).set(process_time * 1e3)

        return response

app.add_middleware(MetricsMiddleware)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Fashion MNIST Prediction API!"}

# Define a predict endpoint

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        contents = await file.read()
        
        # Preprocess the input image
        pixels = preprocess_image(contents)
        
        # Make prediction
        prediction = model.predict(pixels)
        
        # Get the predicted class (index of the highest probability)
        predicted_class = int(np.argmax(prediction))  # Convert numpy.int64 to Python integer
        predicted_label = class_names[predicted_class]
        
        return {"predicted_class": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

