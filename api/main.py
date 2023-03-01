from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2




app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_model/1/cofeedisease.h5")

CLASS_NAMES = ["Cerscospora", "leaf rust", "phoma"]

@app.get("/ping")
async def ping():
    return {"HelloWorld"}

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    resized_image = cv2.resize(image, (256, 256))
    return resized_image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
