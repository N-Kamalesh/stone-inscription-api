import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage.restoration import denoise_wavelet
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import joblib

class TamilInscriptionModel:
    def __init__(self):
        self.model = None
        self.char_list = None
        self.train_dataset = None
        
    def train_and_save(self, base_dir: Path, save_path: str):
        """Train the model and save all necessary components"""
        # Load the original model
        model_path = base_dir / "prediction_files/model_tva_l_v1_2082024.h5"
        self.model = load_model(str(model_path))
        
        # Load and process character mappings
        df = pd.read_csv(base_dir / "prediction_files/recognition-tva2082024.csv", header=0)
        unicode_list = df["Unicode"].tolist()
        
        self.char_list = []
        for element in unicode_list:
            code_list = element.split()
            chars_together = ""
            for code in code_list:
                hex_str = "0x" + code
                char_int = int(hex_str, 16)
                character = chr(char_int)
                chars_together += character
            self.char_list.append(chars_together)
        
        # Save everything using joblib
        model_data = {
            'model': self.model,
            'char_list': self.char_list
        }
        joblib.dump(model_data, save_path)
        
    def load_saved_model(self, model_path: str):
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.char_list = model_data['char_list']
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def preprocess_image(self, image):
        """Preprocess the image for better character recognition"""
        # Resize
        height, width = image.shape[:2]
        scale_percent = 30
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        image = cv2.resize(image, (width, height))

        # Denoise and enhance
        im_visushrink = denoise_wavelet(
            image, convert2ycbcr=False, method='VisuShrink', 
            mode='hard', rescale_sigma=True, wavelet_levels=7, wavelet='coif5'
        )
        im_visushrink = img_as_ubyte(im_visushrink)
        noise = np.std(image)
        dst = cv2.fastNlMeansDenoisingColored(im_visushrink, None, noise / 0.9, noise / 0.9, 7, 21)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        thresh_sauvola = threshold_sauvola(gray, window_size=25)
        binary_sauvola = gray > thresh_sauvola
        sauvola = img_as_ubyte(binary_sauvola)
        sauvola = cv2.bitwise_not(sauvola)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(sauvola, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        
        return erosion

    def segment_image(self, image):
        """Segment the image into individual characters"""
        resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        binary = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours from left to right
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 500]
        sorted_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])
        
        characters = []
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            char_image = resized_image[y:y+h, x:x+w]
            characters.append(char_image)
        
        return characters

    def predict_text(self, characters):
        """Predict text from segmented characters"""
        if not self.model or not self.char_list:
            raise ValueError("Model not loaded")

        predictions = []
        for char_img in characters:
            # Prepare image
            char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            img = cv2.resize(char_img, (60, 60))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            pred = self.model.predict(img, verbose=0)
            predicted_idx = pred.argmax()

            if predicted_idx < len(self.char_list):
                predictions.append(self.char_list[predicted_idx])

        # Combine predictions
        trans = ""
        temp = ""
        for char in predictions:
            if char in ["ெ", "ை"]:
                temp += char
            else:
                trans += temp
                trans += char
                temp = ""
        trans += temp

        return trans

    def save_translation(self, text: str, save_path: Path) -> str:
        """Save the translated text to a file"""
        try:
            # Ensure the directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file with UTF-8 encoding
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            return str(save_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save translation: {str(e)}")

    def process_image(self, image_path: str):
        """Complete pipeline to process an image and return predicted text"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Invalid image file")
        
        # Process
        preprocessed = self.preprocess_image(image)
        characters = self.segment_image(preprocessed)
        if not characters:
            raise ValueError("No characters detected in image")
        
        result = self.predict_text(characters)
        
        return {
            "predicted_text": result,
            "num_characters": len(characters)
        }