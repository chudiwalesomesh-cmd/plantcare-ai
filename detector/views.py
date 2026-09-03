from django.shortcuts import render
from PIL import Image
import numpy as np
import onnxruntime as ort
from pathlib import Path


# Load the lightweight ONNX Plant Disease model
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "plant_model"
    / "onnx"
    / "model_int8.onnx"
)

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


# Plant-wise class mapping
PLANT_CLASSES = {

    "apple": [
        "Apple Scab",
        "Apple with Black Rot",
        "Cedar Apple Rust",
        "Healthy Apple",
    ],

    "blueberry": [
        "Healthy Blueberry Plant",
    ],

    "cherry": [
        "Cherry with Powdery Mildew",
        "Healthy Cherry Plant",
    ],

    "corn": [
        "Corn (Maize) with Cercospora and Gray Leaf Spot",
        "Corn (Maize) with Common Rust",
        "Corn (Maize) with Northern Leaf Blight",
        "Healthy Corn (Maize) Plant",
    ],

    "grape": [
        "Grape with Black Rot",
        "Grape with Esca (Black Measles)",
        "Grape with Isariopsis Leaf Spot",
        "Healthy Grape Plant",
    ],

    "orange": [
        "Orange with Citrus Greening",
    ],

    "peach": [
        "Peach with Bacterial Spot",
        "Healthy Peach Plant",
    ],

    "bell_pepper": [
        "Bell Pepper with Bacterial Spot",
        "Healthy Bell Pepper Plant",
    ],

    "bell pepper": [
        "Bell Pepper with Bacterial Spot",
        "Healthy Bell Pepper Plant",
    ],

    "potato": [
        "Potato with Early Blight",
        "Potato with Late Blight",
        "Healthy Potato Plant",
    ],

    "raspberry": [
        "Healthy Raspberry Plant",
    ],

    "soybean": [
        "Healthy Soybean Plant",
    ],

    "squash": [
        "Squash with Powdery Mildew",
    ],

    "strawberry": [
        "Strawberry with Leaf Scorch",
        "Healthy Strawberry Plant",
    ],

    "tomato": [
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
    ],
}


def home(request):

    if request.method == "POST":

        image = request.FILES.get("leaf_image")
        selected_plant = request.POST.get("plant")

        print("SELECTED PLANT:", selected_plant)

        if image:

            # Open image
            pil_image = Image.open(image).convert("RGB")

            # Resize while maintaining aspect ratio
            width, height = pil_image.size

            if width < height:
                new_width = 256
                new_height = int(height * 256 / width)
            else:
                new_height = 256
                new_width = int(width * 256 / height)

            pil_image = pil_image.resize(
                (new_width, new_height)
            )

            # Center crop to 224x224
            left = (new_width - 224) // 2
            top = (new_height - 224) // 2
            right = left + 224
            bottom = top + 224

            pil_image = pil_image.crop(
                (left, top, right, bottom)
            )

            # Convert image to NumPy array
            image_array = np.array(
                pil_image
            ).astype(np.float32)

            # Rescale 0-255 to 0-1
            image_array = image_array / 255.0

            # Normalize
            image_array = (
                image_array - 0.5
            ) / 0.5

            # HWC -> CHW
            image_array = np.transpose(
                image_array,
                (2, 0, 1)
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Get model input name
            input_name = session.get_inputs()[0].name

            # Run AI prediction
            output = session.run(
                None,
                {input_name: image_array}
            )

            scores = output[0][0]

            # Convert scores to probabilities
            exp_scores = np.exp(
                scores - np.max(scores)
            )

            probabilities = (
                exp_scores / exp_scores.sum()
            )

            # Normalize selected plant name
            plant_key = (
                selected_plant.strip().lower()
                if selected_plant
                else ""
            )

            # Get allowed classes
            allowed_labels = PLANT_CLASSES.get(
                plant_key,
                LABELS
            )

            # Get indexes of allowed classes
            allowed_indices = [
                LABELS.index(label)
                for label in allowed_labels
            ]

            # Get scores only for selected plant
            filtered_scores = probabilities[
                allowed_indices
            ]

            # Normalize filtered probabilities
            if filtered_scores.sum() > 0:
                filtered_scores = (
                    filtered_scores
                    / filtered_scores.sum()
                )

            # Get top predictions
            top_count = min(
                3,
                len(allowed_indices)
            )

            sorted_positions = np.argsort(
                filtered_scores
            )[::-1][:top_count]

            predictions = []

            for position in sorted_positions:

                original_index = (
                    allowed_indices[position]
                )

                predictions.append({
                    "label": LABELS[
                        original_index
                    ],
                    "score": round(
                        float(
                            filtered_scores[position]
                        ) * 100,
                        2
                    )
                })

            return render(
                request,
                "detector/home.html",
                {
                    "predictions": predictions,
                    "selected_plant": selected_plant,
                }
            )

    return render(
        request,
        "detector/home.html"
    )