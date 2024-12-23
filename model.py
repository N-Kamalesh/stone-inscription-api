#model.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage.restoration import denoise_wavelet
from pathlib import Path
import joblib
import imutils
from imutils import contours

class TamilInscriptionModel:
    def __init__(self):
        self.model = None
        self.char_list = None
        self.train_dataset = None
        
    def train_and_save(self, base_dir: Path, save_path: str):
        """Train the model and save all necessary components"""
        try:
            # Load the original model
            model_path = base_dir / "prediction_files/model_dst.h5"
            self.model = load_model(str(model_path))
            
            # Load and process character mappings
            df = pd.read_csv(base_dir / "prediction_files/dst.csv", header=0)
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
        except Exception as e:
            raise RuntimeError(f"Failed to train and save model: {str(e)}")
        
    def load_saved_model(self, model_path: str):
        """Load the saved model and character list"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.char_list = model_data['char_list']
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def preprocess_image(self, image):
        """
        Preprocess the image for better character recognition
        
        Args:
            image: Input BGR image
        Returns:
            Preprocessed grayscale image
        """
        try:
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
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def merge_overlapping_boxes(self, bboxes, y_threshold=5, x_overlap_threshold=5):
        """
        Merges overlapping bounding boxes.

        Args:
            bboxes: List of bounding boxes (x, y, w, h)
            y_threshold: Maximum vertical distance for boxes to be considered overlapping
            x_overlap_threshold: Minimum horizontal overlap for boxes to be merged

        Returns:
            List of merged bounding boxes
        """
        merged_bboxes = []
        bboxes = sorted(bboxes, key=lambda x: x[1])  # Sort by y-coordinate

        while bboxes:
            x, y, w, h = bboxes.pop(0)
            merged = False

            for i, (mx, my, mw, mh) in enumerate(merged_bboxes):
                # Check vertical and horizontal overlap
                vertical_overlap = (y <= my + mh + y_threshold) and (y + h >= my - y_threshold)
                x_overlap = (x + w >= mx and x <= mx + mw)

                if vertical_overlap and x_overlap:
                    # Merge the bounding boxes
                    new_x = min(mx, x)
                    new_y = min(my, y)
                    new_w = max(mx + mw, x + w) - new_x
                    new_h = max(my + mh, y + h) - new_y
                    merged_bboxes[i] = (new_x, new_y, new_w, new_h)
                    merged = True
                    break

            if not merged:
                merged_bboxes.append((x, y, w, h))

        return merged_bboxes

    def segment_image(self, image):
        """
        Segment the image into lines and characters.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            List of lists containing character images for each line
        """
        try:
            # Resize for better detection
            resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            
            # Convert to grayscale if not already
            if len(resized_image.shape) == 3:
                resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            
        # Add Gaussian blur preprocessing
            blurred = cv2.GaussianBlur(resized_image, (7, 7), 0)
            # Edge detection
            edged = cv2.Canny(blurred, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)
            
            # Find contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if not cnts:
                raise ValueError("No contours found in image")
            
            (cnts, _) = contours.sort_contours(cnts)
            
            # Get all bounding boxes
            all_bboxes = []
            for c in cnts:
                if cv2.contourArea(c) < 200:  # Filter small contours
                    continue
                x, y, w, h = cv2.boundingRect(c)
                all_bboxes.append((x, y, w, h))
                
            if not all_bboxes:
                raise ValueError("No valid characters detected")
                
            # Find lines
            min_y = min(all_bboxes, key=lambda x: x[1])[1]
            max_y = max(all_bboxes, key=lambda x: x[1])[1]
            avg_height = sum([bbox[3] for bbox in all_bboxes]) / len(all_bboxes)
            num_lines = max(1, int((max_y - min_y) / avg_height))
            
            # Group boxes by lines
            grouped_lines = {i: [] for i in range(num_lines)}
            for (x, y, w, h) in all_bboxes:
                line_idx = min(num_lines - 1, int((y - min_y) / ((max_y - min_y) / num_lines)))
                grouped_lines[line_idx].append((x, y, w, h))
                
            # Process each line
            characters = []
            for line_idx in sorted(grouped_lines.keys()):
                line = grouped_lines[line_idx]
                if not line:
                    continue
                    
                # Merge overlapping boxes in the line
                merged_line = self.merge_overlapping_boxes(line)
                
                if len(merged_line) <= 1:  # Skip lines with too few characters
                    continue
                    
                # Sort characters left to right
                merged_line = sorted(merged_line, key=lambda x: x[0])
                
                # Extract character images
                line_chars = []
                for x, y, w, h in merged_line:
                    char_img = resized_image[y:y+h, x:x+w]
                    line_chars.append(char_img)
                
                characters.append(line_chars)
                
            return characters
            
        except Exception as e:
            raise ValueError(f"Failed to segment image: {str(e)}")

    
    def predict_text(self, characters):
        """
        Predict text from segmented characters.
        
        Args:
            characters: List of lists containing character images for each line
            
        Returns:
            Predicted text with newlines between lines
        """
        if not self.model or not self.char_list:
            raise ValueError("Model not loaded")
            
        try:
            result = []
            for line in characters:
                line_predictions = []
                for char_img in line:
                    # Prepare image
                    char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
                    img = cv2.resize(char_img, (100, 100))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    # Predict
                    pred = self.model.predict(img, verbose=0)
                    predicted_idx = pred.argmax()
                    
                    if predicted_idx < len(self.char_list):
                        line_predictions.append(self.char_list[predicted_idx])
                
                # Process line predictions
                line_text = ""
                temp = ""
                for char in line_predictions:
                    if char in ["ெ", "ை"]:
                        temp += char
                    else:
                        line_text += temp
                        line_text += char
                        temp = ""
                line_text += temp
                
                result.append(line_text)
                
            return "\n".join(result)
            
        except Exception as e:
            raise ValueError(f"Failed to predict text: {str(e)}")

    def process_image(self, image_path: str):
        """
        Complete pipeline to process an image and return predicted text.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing predicted text and statistics
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image file")
            
            # Process
            preprocessed = self.preprocess_image(image)
            characters = self.segment_image(preprocessed)
            if not characters:
                raise ValueError("No characters detected in image")
            
            result = self.predict_text(characters)
            
            # Calculate statistics
            stats = {
                "predicted_text": result,
                "num_lines": len(characters),
                "chars_per_line": [len(line) for line in characters],
                "total_chars": sum(len(line) for line in characters)
            }
            
            return stats
            
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    def save_translation(self, text: str, save_path: Path) -> str:
        """
        Save the translated text to a file.
        
        Args:
            text: Text to save
            save_path: Path where to save the text
            
        Returns:
            Path to the saved file
        """
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)
            return str(save_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save translation: {str(e)}")