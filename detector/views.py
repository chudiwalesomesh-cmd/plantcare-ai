from django.shortcuts import render
from transformers import pipeline
from PIL import Image


# Load the Plant Disease AI model
classifier = pipeline(
    "image-classification",
    model="kimcomehome/plantvillage-vit-leaf-disease"
)


def home(request):

    if request.method == "POST":

        image = request.FILES.get("leaf_image")
        selected_plant = request.POST.get("plant")

        if image:

            # Convert uploaded file into PIL image
            pil_image = Image.open(image).convert("RGB")

            # Get top 3 AI predictions
            results = classifier(pil_image, top_k=3)

            predictions = []

            for result in results:

                clean_label = result["label"].replace("___", " ").replace("_", " ")

                predictions.append({
                    "label": clean_label,
                    "score": round(result["score"] * 100, 2)
                })

            return render(request, "detector/home.html", {
                "predictions": predictions,
                "selected_plant": selected_plant,
            })

    return render(request, "detector/home.html")