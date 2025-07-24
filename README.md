# 📦 Product Attribute Detection using YOLOv8

This project implements an image-based product detection and classification system using **YOLOv8**, focusing on identifying various consumer products from images. It uses a custom dataset containing multiple product categories and leverages object detection to extract and count product appearances.

---

## 🚀 Project Objective

To train and deploy a YOLOv8 model that can:
- Detect and classify different product types (e.g., Dettol, Maggie, Amul Butter).
- Count the occurrences of each product in an image.
- Visualize bounding boxes and class labels for detected products.

---

## 🗂️ Dataset Structure

The dataset includes 15 product classes, each organized into `train/` and `valid/` folders with YOLO-format `images/` and `labels/`.

### 🏷️ Classes Detected:
- Oppo Enco Buds2
- Nabati Wafer
- Minimalist Sunscreen
- Maggie
- Johnsons Baby Cream
- Himalaya Skin Cream
- Dettol
- Denver Deodorant
- Dabur Honey
- Dabur Chyawanprash
- Citizen Notebook
- Biotique Sunscreen
- Apsara Pencil
- Apis Ginger Garlic Paste
- Amul Butter

---

## 📁 Project Structure

```bash
├── yoo_config.yaml                     # YAML configuration for YOLO
├── /train
│   └── /images, /labels                # Training images & annotations
├── /valid
│   └── /images, /labels                # Validation images & annotations
├── /runs/detect/train/weights/best.pt # Best trained model checkpoint
├── inference_script.ipynb             # End-to-end notebook for training & prediction
