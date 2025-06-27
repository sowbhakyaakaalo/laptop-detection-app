import streamlit as st
import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load ONNX model
model_path = "model_- 21 april 2025 15_58.onnx"
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def detect_objects(img):
    orig = img.copy()
    orig_h, orig_w = orig.shape[:2]

    img_resized = cv2.resize(orig, (640, 640))
    img_input = img_resized.astype(np.float32)
    img_input = np.transpose(img_input, (2, 0, 1)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    outputs = session.run([output_name], {input_name: img_input})
    predictions = outputs[0]

    if predictions.shape[1] < predictions.shape[2]:
        predictions = predictions.transpose(0, 2, 1)
    predictions = predictions[0]

    boxes, scores, class_ids = [], [], []
    threshold = 0.6

    for det in predictions:
        cx, cy, w, h = det[:4]
        class_scores = det[4:]
        score = np.max(class_scores)
        class_id = np.argmax(class_scores)

        if score > threshold:
            x1 = int((cx - w / 2) / 640 * orig_w)
            y1 = int((cy - h / 2) / 640 * orig_h)
            x2 = int((cx + w / 2) / 640 * orig_w)
            y2 = int((cy + h / 2) / 640 * orig_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(class_id)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        class_name = class_names[class_ids[i]]
        conf = scores[i]

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {conf:.2f}"
        cv2.rectangle(orig, (x1, y1 - 20), (x1 + 180, y1), (0, 255, 0), -1)
        cv2.putText(orig, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return orig

# Streamlit UI
st.title("💻 Laptop DIMM Slot Detection")
st.write("Upload a laptop image to detect RAM slots using ONNX model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    result = detect_objects(img_np)
    st.image(result, caption="Detection Result", use_column_width=True)
