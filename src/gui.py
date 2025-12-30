import os
import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox
from tensorflow.keras.models import load_model
import joblib

# Load your pre-trained model
model = load_model(r"models\fake_currency_ann_model.keras")  # Adjust the path as necessary
# Load your PCA model
pca_model = joblib.load(r"models\pca.pkl")  # Adjust the path as necessary

class CurrencyDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Currency Detector")
        self.master.geometry("400x300")

        self.label = Label(master, text="Upload or Capture an Image", font=("Arial", 14))
        self.label.pack(pady=20)

        self.capture_button = Button(master, text="Capture Image", command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.upload_button = Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.result_label = Label(master, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('captured_image.jpg', frame)
            cap.release()
            self.detect_currency('captured_image.jpg')
        else:
            messagebox.showerror("Error", "Failed to capture image.")
            cap.release()

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.detect_currency(file_path)

    def detect_currency(self, image_path):
        # Load and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))  # Resize to match model input
        img = img / 255.0  # Normalize
        img = img.flatten().reshape(1, -1)  # Flatten for PCA

        # Apply PCA transformation
        img_pca = pca_model.transform(img)  # Transform the image using PCA

        # Make prediction
        prediction = model.predict(img_pca)
        currency_type = "Real" if np.argmax(prediction) == 1 else "Fake"  # Adjust based on your model's output

        # Display result
        self.result_label.config(text=f"Result: {currency_type}")

if __name__ == "__main__":
    root = Tk()
    app = CurrencyDetectorApp(root)
    root.mainloop()
