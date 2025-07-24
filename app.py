from flask import Flask, request, render_template
from PIL import Image, ImageDraw
import os
import uuid
import yaml
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO('my_yolo_model.pt')

# Load product names
with open("yoo_config.yaml", "r") as f:
    config = yaml.safe_load(f)
product_list = list(dict.fromkeys(config['names']))
# remove duplicates

@app.route('/')
def index():
    return render_template('index.html', products=product_list)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', products=product_list, error="No image uploaded.")

    file = request.files['image']
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load and predict
    image = Image.open(filepath).convert("RGB")
    results = model(image)
    boxes = results[0].boxes
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    class_names = model.names
    detections = []

    if len(boxes) == 0:
        message = "No objects detected."
    else:
        message = "Detected objects:"
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{class_names[cls_id]} ({conf:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), label, fill="black")
            detections.append(label)

    # Save result image
    result_filename = f"result_{filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    output_image.save(result_path)

    # Return paths for display in HTML
    return render_template('index.html',
                           products=product_list,
                           original_image='/' + filepath,
                           result_image='/' + result_path,
                           detections=detections,
                           message=message)


                           
# ────────────────────── Main ──────────────────────
if __name__ == '__main__':
    app.run(debug=True)
