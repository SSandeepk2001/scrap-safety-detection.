import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image


# --- PAGE SETTINGS ---
st.set_page_config(page_title="🛡️ Scrap Safety Detection", page_icon="⚙️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    h1 {
        text-align: center;
        color: #003366;
    }
    .sidebar .sidebar-content {
        background-color: #dbe9f4;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD YOLOv8 MODEL ---
model_path = os.path.join("best.pt")
model = YOLO(model_path)

# --- SIDEBAR ---
st.sidebar.title("📊 Dashboard Panel")
st.sidebar.markdown("Welcome to the **Scrap-Based Liquid Steel Safety System**. 🚧")
st.sidebar.markdown("---")
st.sidebar.subheader("🛠 Project Information")
st.sidebar.markdown("""
- **Project**: Hazard Detection
- **Technology**: YOLOv8, Streamlit
- **Goal**: Enhance Safety via Real-Time Detection
""")
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Target KPIs")
st.sidebar.markdown("""
- 🎯 Accuracy > 90%
- 🚨 Response time < 2 sec
- 🛡️ Reduce manual inspection by 50%
""")
st.sidebar.markdown("---")
st.sidebar.info("📤 Upload a file from the main page to start detection!")

# --- MAIN HEADER ---
st.title("🛡️ Enhancing Safety and Hazard Management in Scrap-Based Liquid Steel Production")
st.markdown("---")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("📤 Upload a Video or Image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

# --- PROCESSING ---
if uploaded_file is not None:
    confidence_scores = []
    class_labels = []

    if uploaded_file.type.startswith("image"):
        st.subheader("📷 Detected Hazardous Materials (Image)")
        with st.spinner('Detecting hazards in the image... 🔍'):
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            results = model(image_np)[0]

            for result in results.boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                conf = result.conf[0]
                cls = int(result.cls[0])
                confidence_scores.append(float(conf))
                class_labels.append(model.names[cls])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            st.image(image_np, channels="BGR", use_column_width=True)
            st.success(f"✅ Total Hazards Detected: {len(results.boxes)}")

    elif uploaded_file.type.startswith("video"):
        st.subheader("🎥 Detected Hazardous Materials (Video)")
        with st.spinner('Analyzing video frames... ⏳'):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            temp_filename = tfile.name

            cap = cv2.VideoCapture(temp_filename)
            stframe = st.empty()

            frame_count = 0
            hazard_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                results = model(frame)[0]

                for result in results.boxes:
                    hazard_count += 1
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    conf = result.conf[0]
                    cls = int(result.cls[0])
                    confidence_scores.append(float(conf))
                    class_labels.append(model.names[cls])
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                stframe.image(frame, channels="BGR", use_column_width=True)

            cap.release()
            os.remove(temp_filename)

            st.success(f"✅ Total Frames Processed: {frame_count}")
            st.success(f"✅ Total Hazards Detected: {hazard_count}")

    st.balloons()

    # --- DATA VISUALIZATION ---
    if confidence_scores:
        st.markdown("---")
        st.header("📈 Detection Insights")

        avg_confidence = np.mean(confidence_scores) * 100
        st.metric("🎯 Detection Accuracy (Avg Confidence)", f"{avg_confidence:.2f}%")

        st.subheader("🔎 Confidence Score Distribution")
        fig, ax = plt.subplots()
        ax.hist(confidence_scores, bins=10, color='#0077b6', edgecolor='black')
        ax.set_title("Confidence Scores Histogram")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Number of Detections")
        st.pyplot(fig)

        st.subheader("📊 Class-wise Detection Count")
        if class_labels:
            unique, counts = np.unique(class_labels, return_counts=True)
            fig2, ax2 = plt.subplots()
            ax2.bar(unique, counts, color='#00b4d8', edgecolor='black')
            ax2.set_title("Class-wise Detection Count")
            ax2.set_xlabel("Detected Classes")
            ax2.set_ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

st.markdown("---")
st.info("📢 Upload a video or image file to see hazardous material detection in real-time!")
