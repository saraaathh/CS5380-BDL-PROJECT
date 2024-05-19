from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import io

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

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Fashion MNIST Prediction API!"}

# Define a predict endpoint
@app.post("/predict")
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
        prediction = class_names[predicted_class]
        return {"predicted_class": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

