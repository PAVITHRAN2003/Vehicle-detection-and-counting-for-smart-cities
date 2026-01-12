# 🚦 Smart City Vehicle Detection & Counting System

An end-to-end **computer vision system** for detecting and counting vehicles in urban traffic scenes using **YOLOv8** and a **Streamlit-based interactive dashboard**.  
Designed for **smart city analytics**, traffic monitoring, and real-time decision support.

---

## 🚀 Why This Project Matters

Urban traffic systems generate massive video data, but extracting **actionable insights** from it remains challenging.

This project demonstrates how **deep learning + modern ML tooling** can be used to:
- Detect vehicles in real-world traffic scenes
- Classify vehicle types (car, bus, van, others)
- Provide real-time visual analytics through a web interface
- Lay the foundation for scalable smart-city deployments

---

## 🧠 Key Features

- ✅ YOLOv8-based vehicle detection
- ✅ Supports multiple vehicle classes
- ✅ Real-time inference visualization
- ✅ Streamlit web dashboard for demos & presentations
- ✅ Clean, modular project structure
- ✅ Easily deployable (local / cloud / container-ready)

---

## 🏗️ System Architecture

Input Image / Video
↓
YOLOv8 Detection Model
↓
Bounding Boxes + Class Predictions
↓
Post-processing & Counting
↓
Streamlit Dashboard (Visualization + Metrics)


---

## 🧪 Model & Dataset

- **Model**: YOLOv8 (Ultralytics)
- **Pretrained Base**: `yolov8s.pt`
- **Custom Classes**:
  - Car
  - Bus
  - Van
  - Others
- **Dataset**: UA-DETRAC (urban traffic surveillance)

> ⚠️ Note: Due to limited training epochs (Colab constraints), some class confusion (e.g., car vs bus) may occur.

## 🏗️ System Architecture & Design

The system processes input video feeds through a preprocessing pipeline before passing them to the YOLOv8 model for inference. Results are post-processed to track counts and displayed on the dashboard.

```text
┌───────────────────────────────┐
│     Traffic Image / Video     │
│   (CCTV / Drone / Dataset)    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Preprocessing Layer      │
│    Resize • Normalize • IO    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│    YOLOv8 Detection Model     │
│   (Fine-tuned on Traffic)     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│    Post-Processing Engine     │
│  NMS • Thresholding • Count   │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│   Analytics & Visualization   │
│      Streamlit Dashboard      │
│     Metrics • Charts • UI     │
└───────────────────────────────┘

```

## 🖥️ Tech Stack

**AI / ML**
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV

**Web & Deployment**
- Streamlit
- Python
- Docker-ready structure

**Tools**
- Google Colab (training)
- GitHub
- VS Code

---

## ⚙️ Installation & Usage

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/Vehicle-detection-and-counting-for-smart-cities.git
cd Vehicle-detection-and-counting-for-smart-cities


2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the App
bash
Copy code
streamlit run app.py
📈 Future Improvements
Vehicle tracking with unique IDs (DeepSORT / ByteTrack)

Improved class balance and longer training

Video stream support

Traffic density & congestion metrics

Cloud deployment (AWS / GCP)

👨‍💻 Author
Pavithran Gnanasekaran
MS in Computer Science (AI & ML) — University at Buffalo

GitHub: https://github.com/Pavithran

LinkedIn: https://linkedin.com/in/Pavithran

