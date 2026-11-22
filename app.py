import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Violation Analytics", layout="wide")
st.title("Smart Traffic & Helmet / License Plate Violation Analytics")

@st.cache_resource
def load_models():
    model = YOLO("weights/best.pt")
    reader = easyocr.Reader(["en"], gpu=False)
    return model, reader

model, ocr_reader = load_models()

def is_inside_or_overlapping(parent_box, child_box, pad_top=40):
    px1, py1, px2, py2 = parent_box[0], max(0, parent_box[1] - pad_top), parent_box[2], parent_box[3]
    cx_mid = (child_box[0] + child_box[2]) / 2
    cy_mid = (child_box[1] + child_box[3]) / 2
    return (px1 <= cx_mid <= px2) and (py1 <= cy_mid <= py2)

conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.20, 0.85, 0.35, 0.05)
tab1, tab2 = st.tabs(["Live Image Detection & OCR", "Dataset Benchmark Analytics"])

with tab1:
    uploaded = st.file_uploader("Upload a traffic image (.jpg, .png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        annotated = img.copy()

        pred = model.predict(img, conf=conf_thresh, verbose=False)[0]
        names = model.names
        bikes, no_helmets, plates = [], [], []

        for box in pred.boxes:
            label = names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 3)

            if label == "bike":
                bikes.append((x1, y1, x2, y2, conf))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
            elif label == "helmet":
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(annotated, f"Helmet {conf}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            elif label == "no-helmet":
                no_helmets.append((x1, y1, x2, y2, conf))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"NO HELMET {conf}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            elif label == "number-plate":
                plates.append((x1, y1, x2, y2, conf))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)

        violations = []
        for idx, nh in enumerate(no_helmets, 1):
            linked_bike = next((b for b in bikes if is_inside_or_overlapping(b, nh)), None)
            linked_plate = None
            if linked_bike:
                linked_plate = next((p for p in plates if is_inside_or_overlapping(linked_bike, p, pad_top=0)), None)

            plate_str = "UNREADABLE"
            if linked_plate:
                px1, py1, px2, py2, _ = linked_plate
                crop = img[max(0, py1):py2, max(0, px1):px2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    raw = ocr_reader.readtext(gray, detail=0)
                    cleaned = "".join([c for c in "".join(raw) if c.isalnum()]).upper()
                    if len(cleaned) >= 4:
                        plate_str = cleaned

            violations.append({
                "Violation #": idx,
                "Type": "No Helmet",
                "Confidence": nh[4],
                "Extracted Plate": plate_str
            })

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Model Inference Output", use_container_width=True)
        with col2:
            st.subheader("Detected Violations")
            if violations:
                st.dataframe(pd.DataFrame(violations), use_container_width=True)
            else:
                st.success("No helmet violations detected in this frame.")

with tab2:
    st.subheader("Training Curves & Confusion Matrix")
    c1, c2 = st.columns(2)
    c1.image("outputs/results.png", caption="Training Loss & mAP Metrics")
    c2.image("outputs/confusion_matrix.png", caption="4-Class Confusion Matrix")
    st.subheader("Bulk Dataset Analytics")
    st.image("outputs/final_traffic_analytics.png", caption="Dataset-Wide Compliance & OCR Breakdown")
    st.subheader("Extracted Violation Records Sample")
    df_sample = pd.read_csv("outputs/violation_analytics_dataset.csv")
    st.dataframe(df_sample.head(25), use_container_width=True)
