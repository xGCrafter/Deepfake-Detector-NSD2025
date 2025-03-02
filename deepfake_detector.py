import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QMovie
from PyQt5.QtCore import Qt, QTimer
from PIL import Image, ImageGrab
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import face_recognition

# Define model architecture (ensure this matches the one used during training)
def build_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Create model and load weights
model_path = "/Users/gauravagrawal/Desktop/instagram_ai_deepfake/dffnetv2B0.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit("Missing model file: dffnetv2B0.h5")

model = build_model()
model.load_weights(model_path, by_name=True, skip_mismatch=True)

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize image to match Xception input
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define deepfake detection function
def detect_deepfake(image_path):
    img = Image.open(image_path).convert('RGB')  # Open and convert image
    img = preprocess_image(img)  # Preprocess the image
    prediction = model(img, training=False).numpy()[0][0]  # Get prediction
    return "DEEPFAKE DETECTED NOOOOOOOOOOO" if prediction > 0.5 else "REAL IMAGE"

# Take a screenshot, crop face, and analyze it
def take_screenshot_and_detect():
    screenshot_path = "screenshot.png"
    
    # Remove the previous screenshot if it exists
    if os.path.exists(screenshot_path):
        os.remove(screenshot_path)

    try:
        screenshot = ImageGrab.grab()  # Capture the screen
        screenshot.save(screenshot_path)  # Save it
        
        if not os.path.exists(screenshot_path):
            print(f"Error: Screenshot not found at {screenshot_path}!")
            return "Screenshot capture failed!"

        print(f"Screenshot saved at: {os.path.abspath(screenshot_path)}")
        
        # Detect face and crop
        image = face_recognition.load_image_file(screenshot_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            return "No face detected!"

        # Get first detected face
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]  # Crop the face

        # Convert to PIL and save cropped face
        face_pil = Image.fromarray(face_image)
        face_pil.save("face_crop.png")

        # Run detection on the cropped face
        result = detect_deepfake("face_crop.png")
        return result

    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return "Screenshot failed!"

# ------ UI -------
class DeepFakeDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.gradient_step = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gradient)
        self.timer.start(200)  # Update every 200ms
        self.max_steps = 10  # Limit the number of steps to stop looping

    def initUI(self):
        self.setWindowTitle("Deepfake Detector")  # Set window title
        self.setGeometry(100, 60, 400, 120) 
        
        layout = QVBoxLayout()
        
        # Button
        self.button = QPushButton("DETECT")
        self.button.setStyleSheet(
            "font-size: 20px; background-color: #FF474C; color: white; border: none; "
            "padding: 15px; border-radius: 20px; font-weight: bold; box-shadow: 2px 2px 10px rgba(0,0,0,0.3);"
        )
        self.button.setFixedSize(180, 50)  # Make button smaller
        self.button.clicked.connect(self.run_detection)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Add QLabel for GIF (this will be used only when deepfake is detected)
        self.gif_label = QLabel(self)
        self.gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gif_label)

        self.setLayout(layout)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        colors = ["#FF474C", "#D43F3A", "#A8332E", "#7F2823", "#4F1A16"]
        stripe_height = self.height() // len(colors)
        
        for i, color in enumerate(colors[:self.gradient_step + 1]):
            painter.fillRect(0, i * stripe_height, self.width(), stripe_height, QColor(color))
    
    def update_gradient(self):
        if self.gradient_step < self.max_steps - 1:
            self.gradient_step += 1
            self.update()
        else:
            self.timer.stop()

    def run_detection(self):
        # Start the detection
        result = take_screenshot_and_detect()

        # Check if it's a deepfake and show the GIF accordingly
        if "DEEPFAKE DETECTED" in result:
            self.start_loading_gif()
        
        # Show the result in a popup (this includes the GIF for deepfakes)
        QMessageBox.information(self, "Detection Result", result)

        if "DEEPFAKE DETECTED" in result:
            self.stop_loading_gif()

    def start_loading_gif(self):
        # Load and start the GIF
        self.movie = QMovie("/Users/gauravagrawal/Desktop/instagram_ai_deepfake/siren.gif") 
        self.gif_label.setMovie(self.movie)
        self.movie.start()

    def stop_loading_gif(self):
        # Stop the GIF after showing result
        if self.movie:
            self.movie.stop()
            self.gif_label.clear()


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepFakeDetectorApp()
    window.show()
    sys.exit(app.exec())
