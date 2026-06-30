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
    """Preprocesses the image to match the model's expected input.

    The target resolution is read directly from the TFLite model's input tensor
    so this function is always correct regardless of how the model was trained.
    Expected tensor shape: [1, H, W, 3]
    """
    _, target_h, target_w, _ = input_details[0]['shape']   # e.g. [1, 256, 256, 3]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((target_w, target_h))              # PIL uses (width, height)

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)           # → (1, H, W, 3)

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
        
        # ── Classification threshold ──────────────────────────────────
        # 0.35 = more sensitive (catches more fakes, may flag some real images)
        # 0.50 = balanced (default, higher false-negative rate)
        # Tune this value based on your validation results.
        THRESHOLD = 0.35

        score = float(prediction[0][0])
        is_deepfake = bool(score > THRESHOLD)

        return {
            "filename":    file.filename,
            "is_deepfake": is_deepfake,
            "raw_score":   round(score, 6),          # 0 = real, 1 = fake
            "confidence":  score if is_deepfake else 1 - score,
            "threshold":   THRESHOLD,
        }
    except Exception as e:
        return {"error": str(e)}