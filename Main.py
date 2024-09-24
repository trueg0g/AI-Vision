import sys
import torch
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 small model

class ImageDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("GoG AI")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.label = QLabel("Click the button to select an image.")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.label)

        self.button = QPushButton("Select Image")
        self.button.clicked.connect(self.select_image)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            self.detect_objects(file_path)

    def detect_objects(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image!")
            return

        results = model(img)

        self.show_detections(results)

    def show_detections(self, results):

        results_img = np.array(results.render()[0])
        results_img = cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB)

        h, w, ch = results_img.shape
        bytes_per_line = ch * w
        q_img = QImage(results_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)

        self.label.setPixmap(pixmap.scaled(800, 600, QtCore.Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageDetectionApp()
    window.show()
    sys.exit(app.exec_())
