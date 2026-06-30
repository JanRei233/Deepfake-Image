from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Deepfake Detection API")

# Enable CORS so your frontend can communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="deepfake_efficientnet_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def prepare_image(image_bytes):
    """Preprocesses the image to match the model's expected input."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # IMPORTANT: Change 224 to match the size used in your Jupyter Notebook
    image = image.resize((224, 224)) 
    
    img_array = np.array(image, dtype=np.float32)
    # Normalize if you did this during training (e.g., dividing by 255.0)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        
        # Preprocess
        input_data = prepare_image(image_bytes)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Extract the score (assuming binary classification where > 0.5 is Fake)
        score = float(prediction[0][0])
        is_deepfake = bool(score > 0.5)
        
        return {
            "filename": file.filename,
            "is_deepfake": is_deepfake,
            "confidence": score if is_deepfake else 1 - score
        }
    except Exception as e:
        return {"error": str(e)}