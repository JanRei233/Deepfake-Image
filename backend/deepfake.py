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

    IMPORTANT — EfficientNet preprocessing:
      Training used tf.keras.applications.efficientnet.preprocess_input()
      which maps pixel values from [0, 255] to [-1, 1].
      Using /255.0 (simple normalisation) produces the WRONG input distribution
      and is the primary cause of poor inference accuracy.

    The target resolution is read directly from the TFLite model's input tensor
    so this function is always correct regardless of how the model was trained.
    Expected tensor shape: [1, H, W, 3]
    """
    _, target_h, target_w, _ = input_details[0]['shape']   # e.g. [1, 256, 256, 3]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((int(target_w), int(target_h)))    # PIL uses (width, height)

    img_array = np.array(image, dtype=np.float32)

    # EfficientNet preprocessing: scale [0,255] to [-1, 1]
    # Equivalent to: tf.keras.applications.efficientnet.preprocess_input()
    img_array = (img_array / 127.5) - 1.0

    img_array = np.expand_dims(img_array, axis=0)           # shape: (1, H, W, 3)

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

        # -- Classification threshold -----------------------------------------
        # 0.35 = more sensitive (catches more fakes, may flag some real images)
        # 0.50 = balanced (default, higher false-negative rate)
        # After retraining, update this to the best_threshold printed by train.py
        THRESHOLD = 0.45

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