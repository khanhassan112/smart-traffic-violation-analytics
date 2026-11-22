# Smart Traffic & Helmet / License Plate Violation Analytics

An end-to-end computer vision and traffic analytics pipeline built with **YOLOv8** and **EasyOCR** to detect two-wheeler helmet violations, associate violating riders with their parent motorcycle bounding box, and extract license plate numbers.

## System Pipeline
1. **Multi-Class Object Detection (`YOLOv8`):** Trained on 20,000+ annotated traffic images across 4 classes: `bike`, `helmet`, `no-helmet`, and `number-plate`.
2. **Geometric Rider-to-Plate Association:** Maps `no-helmet` detections to their enclosing `bike` bounding box and isolates the corresponding `number-plate` region to prevent cross-vehicle plate mismatches.
3. **Automated OCR & Logging (`EasyOCR`):** Extracts alphanumeric registration numbers from violating vehicles and exports structured violation records (`outputs/violation_analytics_dataset.csv`) alongside cropped evidence frames.
4. **Interactive Analytics Dashboard (`Streamlit`):** Provides real-time image inference and dataset-wide compliance visualizations.

## Project Structure
- `app.py` — Streamlit web application for live inference and analytics visualization
- `traffic_violation_pipeline.ipynb` — Complete Google Colab training, evaluation, and bulk OCR notebook
- `weights/best.pt` — Custom-trained YOLOv8 weights
- `outputs/` — Training curves, confusion matrix, analytics figures, evidence crops, and extracted CSV logs

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
