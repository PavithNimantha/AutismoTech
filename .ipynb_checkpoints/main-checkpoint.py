from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uvicorn

# Define constants
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Same as model input size
MODEL_PATH = "fine_tuned_model.h5"  # Path to your trained model

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = {
    0: "asd_mild_coloring_3-6",
    1: "asd_moderate_coloring_3-6",
    2: "asd_severe_coloring_3-6",
    3: "asdwithcd_mild_coloring_3-6",
    4: "asdwithcd_moderate_coloring_3-6",
    5: "asdwithcd_severe_coloring_3-6",
    6: "non_asd_normal_coloring_3-6"
}

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ASD Classification API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        
        # Load and preprocess the image
        img = load_img(
            path_or_stream=contents,
            target_size=(IMG_HEIGHT, IMG_WIDTH)
        )
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)