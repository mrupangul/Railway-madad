import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Load custom YOLOv5 model with your own trained weights (e.g., 'best.pt')
custom_weights_path = "C:\\Users\\avant\\PycharmProjects\\ml\\yolov5\\runs\\train\\exp7\\weights\\best.pt"
yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_weights_path, force_reload=True)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the complaint categories and departments (customize these based on your dataset)
complaint_categories = [
    "Air_Conditioner", "Alcohol_Narcotics", "Broken_Window_Seat", "Charging_Points",
    "DirtyTorn", "Eve-teasing_Misbehavior_Rape", "Fan", "Harassment_Extortion",
    "Interior_Cleanliness", "Lights", "Medical_Assistance", "Nuisance_Hawkers_Beggars",
    "Smoking", "Smoking_Drinking", "Tap_Leaking", "Theft_Snatching", "Toilet",
    "Unauthorized_Person_in_Ladies", "Washbasin"
]

departments = {
    "Air_Conditioner": "Electrical Department",
    "Alcohol_Narcotics": "RPF Department",
    "Broken_Window_Seat": "CNW/Engineering Department",
    "Charging_Points": "Electrical Department",
    "DirtyTorn": "CNW Department",
    "Eve-teasing_Misbehavior_Rape": "RPF Department",
    "Fan": "Electrical Department",
    "Harassment_Extortion": "RPF Department",
    "Interior_Cleanliness": "CNW Department",
    "Lights": "Electrical Department",
    "Medical_Assistance": "Commercial Department",
    "Nuisance_Hawkers_Beggars": "RPF Department",
    "Smoking": "RPF Department",
    "Smoking_Drinking": "RPF Department",
    "Tap_Leaking": "CNW/Engineering Department",
    "Theft_Snatching": "RPF Department",
    "Toilet": "CNW Department",
    "Unauthorized_Person_in_Ladies": "RPF Department",
    "Washbasin": "CNW Department"
}

# Initialize the dictionary to accumulate complaints for each department
department_complaints = {
    "Electrical Department": [],
    "RPF Department": [],
    "CNW Department": [],
    "Engineering Department": [],
    "Commercial Department": []
}

# Initialize a counter for sequential numbering
image_counter = 1


# Function to categorize and map complaints to departments
def categorize_complaint(image_path):
    global image_counter

    # Generate the new filename with ascending number
    base_dir, original_name = os.path.split(image_path)
    file_ext = os.path.splitext(original_name)[1]
    new_filename = f"{str(image_counter).zfill(3)}{file_ext}"
    new_image_path = os.path.join(base_dir, new_filename)

    # Rename the file
    os.rename(image_path, new_image_path)

    # YOLOv5 object detection
    results = yolo(new_image_path)
    detected_objects = results.pandas().xyxy[0]['name'].tolist()

    # Convert image to PIL for CLIP processing
    image = Image.open(new_image_path)
    inputs = clip_processor(text=complaint_categories, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    # Get the category with the highest similarity score
    probs = outputs.logits_per_image.softmax(dim=1)
    category_index = probs.argmax().item()
    category = complaint_categories[category_index]

    # Map category to department
    department = departments.get(category, "Unknown")

    # Append the complaint to the corresponding department
    if department != "Unknown":
        department_complaints[department].append((new_image_path, category))

    # Increment the image counter
    image_counter += 1

    return category, department, detected_objects


# Route for the home page with upload form
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Save the uploaded file
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the file to a temporary location
        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        # Process the uploaded image
        category, department, detected_objects = categorize_complaint(image_path)

        # Redirect to results page
        return redirect(url_for("results", filename=file.filename))

    return render_template("index.html")


# Route to display results
@app.route("/results/<filename>")
def results(filename):
    image_path = os.path.join("static", filename)

    # Display the accumulated complaints after processing
    return render_template("results.html",
                           image_path=image_path,
                           department_complaints=department_complaints)


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)