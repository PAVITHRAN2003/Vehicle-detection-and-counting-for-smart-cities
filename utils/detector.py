from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, conf=0.25)
        annotated = results[0].plot()
        classes = results[0].boxes.cls.tolist()
        return annotated, classes
