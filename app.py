import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import cv2
import numpy as np
from utils.detector import VehicleDetector


# ----------------------------
# Page Config & Title
# ----------------------------
st.set_page_config(
    page_title="Smart City Vehicle Analytics",
    page_icon="🚦",
    layout="wide"
)

st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="font-size:2.5rem;">🚦 Vehicle Detection & Analytics for Smart Cities</h1>
        <p style="font-size:1rem;color:gray;">
        Leveraging YOLOv8 for real-time traffic analysis and AI-driven urban mobility insights.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# ----------------------------
# Model Loading with Cache
# ----------------------------
@st.cache_resource
def load_model():
    return VehicleDetector("models/best.pt")

detector = load_model()

# ----------------------------
# Sidebar: Info & Dataset
# ----------------------------
with st.sidebar:
    st.header("📚 About This Project")
    st.write("""
    - **Dataset:** Traffic images sourced from multiple urban areas, covering vehicles like cars, trucks, buses, and motorcycles.
    - **YOLOv8 Pipeline:** Utilizes a state-of-the-art one-stage object detection model for high-speed, accurate detection.
    - **Training Constraints:** Currently trained for **5 epochs** due to limited compute resources. Optimal performance requires **≥30 epochs**.
    - **Use Cases:**  
        - Smart traffic monitoring  
        - AI-powered urban planning  
        - Real-time congestion alerts  
        - Autonomous vehicle insights
    - **Future Work:**  
        - Multi-camera video integration  
        - Real-time analytics dashboard  
        - Cloud deployment for city-scale monitoring
    """)

# ----------------------------
# File Uploader
# ----------------------------
st.subheader("📂 Upload Traffic Image")
uploaded_file = st.file_uploader(
    "Upload an image of traffic (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    bytes_data = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # ----------------------------
    # Detection & Output
    # ----------------------------
    with st.spinner("Running YOLOv8 vehicle detection..."):
        output, classes = detector.detect(image)

    st.image(output, channels="BGR", use_column_width=True)

    st.markdown(
        f"""
        <div style="margin-top:20px;">
            <h3>📊 Detection Summary</h3>
            <p style="font-size:1.1rem;">
            Total Vehicles Detected: <strong>{len(classes)}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True
    )

# ----------------------------
# Footer: Highlight Tech
# ----------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center;color:gray;font-size:0.9rem;">
    Built with <strong>Python, Streamlit, OpenCV</strong> and <strong>YOLOv8</strong> • Real-time AI for Smart City Mobility
    </p>
    """, unsafe_allow_html=True
)
