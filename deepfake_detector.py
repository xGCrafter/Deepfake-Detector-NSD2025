import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageGrab
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt6.QtGui import QPainter, QColor, QRegion
from PyQt6.QtCore import Qt
import os
from efficientnet_pytorch import EfficientNet

# Define the deepfake model class
class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        # Load a pre-trained EfficientNet model
        self.model = EfficientNet.from_pretrained("efficientnet-b0")
        # Modify the final layer to output a single value (probability of being a deepfake)
        self.model._fc = nn.Linear(self.model._fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# Initialize model
model = DeepFakeModel()
model.eval()  # Set the model to evaluation mode

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def preprocess_image(image_path):
    """Preprocesses the image for the model."""
    img = Image.open(image_path).convert('RGB')  # Open and convert image to RGB
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

def detect_deepfake(image_path):
    """Predicts whether the given image is a deepfake or real."""
    img = preprocess_image(image_path)  # Preprocess the image
    with torch.no_grad():  # Disable gradient calculation
        output = model(img).sigmoid().item()  # Get model output and apply sigmoid
    return "DEEPFAKE DETECTED NOOOOOOOOOOO" if output > 0.5 else "REAL IMAGE"  # Thresholding

def take_screenshot_and_detect():
    """Captures a screenshot using ImageGrab and runs deepfake detection."""
    screenshot_path = "screenshot.png"
    try:
        screenshot = ImageGrab.grab()  # Capture screenshot
        screenshot.save(screenshot_path)  # Save screenshot
        
        if not os.path.exists(screenshot_path):
            print(f"Error: Screenshot not found at {screenshot_path}!")
            return "Screenshot capture failed!"

        print(f"Screenshot saved successfully at: {os.path.abspath(screenshot_path)}")
        
        result = detect_deepfake(screenshot_path)  # Detect deepfake in the screenshot
        return result
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return "Screenshot failed!"

class DeepFakeDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Deepfake Detector")  # Set window title
        self.setGeometry(100, 100, 300, 300)  # Set window size (square for circular effect)

        # Create a circular mask for the window
        self.setMask(QRegion(0, 0, 300, 300, QRegion.RegionType.Ellipse))  # Set circular shape
        self.button = QPushButton("DETECT", self)  # Create button
        self.button.setGeometry(10, 10, 280, 280)  # Set button size and position
        self.button.setStyleSheet("font-size: 20px; background-color: #FF474C; border: none; border-radius: 140px;")  # Style button with larger text
        self.button.clicked.connect(self.run_detection)  # Connect button click to detection method

    def run_detection(self):
        result = take_screenshot_and_detect()  # Run detection
        QMessageBox.information(self, "Detection Result", result)  # Show result in message box

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create application
    window = DeepFakeDetectorApp()  # Create main window
    window.show()  # Show the window
    sys.exit(app.exec())  # Execute the application