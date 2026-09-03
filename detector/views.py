from django.shortcuts import render
from PIL import Image
import numpy as np
import onnxruntime as ort
from pathlib import Path


# Load the lightweight ONNX Plant Disease model
MODEL_PATH = Path(__file__).resolve().parent.parent / "plant_model" / "onnx" / "model_int8.onnx"

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)


# 38 disease classes
LABELS = [
    "Apple Scab",
    "Apple with Black Rot",
    "Cedar Apple Rust",
    "Healthy Apple",
    "Healthy Blueberry Plant",
    "Cherry with Powdery Mildew",
    "Healthy Cherry Plant",
    "Corn (Maize) with Cercospora and Gray Leaf Spot",
    "Corn (Maize) with Common Rust",
    "Corn (Maize) with Northern Leaf Blight",
    "Healthy Corn (Maize) Plant",
    "Grape with Black Rot",
    "Grape with Esca (Black Measles)",
    "Grape with Isariopsis Leaf Spot",
    "Healthy Grape Plant",
    "Orange with Citrus Greening",
    "Peach with Bacterial Spot",
    "Healthy Peach Plant",
    "Bell Pepper with Bacterial Spot",
    "Healthy Bell Pepper Plant",
    "Potato with Early Blight",
    "Potato with Late Blight",
    "Healthy Potato Plant",
    "Healthy Raspberry Plant",
    "Healthy Soybean Plant",
    "Squash with Powdery Mildew",
    "Strawberry with Leaf Scorch",
    "Healthy Strawberry Plant",
    "Tomato with Bacterial Spot",
    "Tomato with Early Blight",
    "Tomato with Late Blight",
    "Tomato with Leaf Mold",
    "Tomato with Septoria Leaf Spot",
    "Tomato with Spider Mites or Two-spotted Spider Mite",
    "Tomato with Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy Tomato Plant",
]


def home(request):

    if request.method == "POST":

        image = request.FILES.get("leaf_image")
        selected_plant = request.POST.get("plant")

        if image:

            # Open and resize image
            pil_image = Image.open(image).convert("RGB")
            pil_image = pil_image.resize((224, 224))

            # Convert image to NumPy array
            image_array = np.array(pil_image).astype(np.float32)

            # Normalize image
            image_array = image_array / 255.0

            # Change format: HWC -> CHW
            image_array = np.transpose(image_array, (2, 0, 1))

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Get model input name
            input_name = session.get_inputs()[0].name

            # Run AI prediction
            output = session.run(None, {input_name: image_array})

            scores = output[0][0]

            # Convert scores to probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / exp_scores.sum()

            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[::-1][:3]

            predictions = []

            for index in top_indices:

                predictions.append({
                    "label": LABELS[index],
                    "score": round(float(probabilities[index]) * 100, 2)
                })

            return render(request, "detector/home.html", {
                "predictions": predictions,
                "selected_plant": selected_plant,
            })

    return render(request, "detector/home.html")