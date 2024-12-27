from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from skimage import img_as_ubyte
from skimage.filters import threshold_sauvola
from skimage.restoration import denoise_wavelet
import joblib
import imutils
from imutils import contours
import shutil
import os
import matplotlib.pyplot as plt

class TamilInscriptionModel:
    def __init__(self):
        self.model = None
        self.char_list = None
        self.train_dataset = None
        self.debug_mode = False
        self.debug_dir = None
        
    def set_debug_mode(self, debug_mode: bool, debug_dir: Path = None):
        """Enable or disable debug mode and set debug directory"""
        self.debug_mode = debug_mode
        if debug_mode and debug_dir:
            self.debug_dir = debug_dir
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def save_debug_image(self, image, name: str):
        """Save intermediate images when debug mode is enabled"""
        if self.debug_mode and self.debug_dir:
            save_path = self.debug_dir / f"{name}.png"
            cv2.imwrite(str(save_path), image)
            return save_path
        return None

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
        """Load the saved model with proper character mapping initialization"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.char_list = model_data['char_list']
            
            # Initialize training dataset for class indices
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
            self.train_dataset = train_datagen.flow_from_directory(
                os.path.join(os.path.dirname(model_path), '../prediction_files/train'),
                target_size=(100, 100),
                batch_size=32,
                class_mode='categorical'
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def resize_image(self, image, scale_percent=30):
        """Resize image maintaining aspect ratio"""
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def preprocess_image(self, image):
        """Enhanced preprocessing pipeline with debug saves"""
        try:
            # Resize
            resized = self.resize_image(image, scale_percent=30)
            if self.debug_mode:
                self.save_debug_image(resized, "01_resized")

            # Wavelet denoising
            im_visushrink = denoise_wavelet(
                resized, convert2ycbcr=False, method='VisuShrink',
                mode='hard', rescale_sigma=True, wavelet_levels=7, wavelet='coif5'
            )
            im_visushrink = img_as_ubyte(im_visushrink)
            if self.debug_mode:
                self.save_debug_image(im_visushrink, "02_wavelet_denoised")

            # Non-local means denoising
            noise = np.std(resized)
            dst = cv2.fastNlMeansDenoisingColored(im_visushrink, None, noise / 0.9, noise / 0.9, 7, 21)
            if self.debug_mode:
                self.save_debug_image(dst, "03_nlm_denoised")

            # Convert to grayscale
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            if self.debug_mode:
                self.save_debug_image(gray, "04_grayscale")

            # Sauvola thresholding
            thresh_sauvola = threshold_sauvola(gray, window_size=25)
            binary_sauvola = gray > thresh_sauvola
            sauvola = img_as_ubyte(binary_sauvola)
            sauvola = cv2.bitwise_not(sauvola)
            if self.debug_mode:
                self.save_debug_image(sauvola, "05_thresholded")

            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(sauvola, kernel, iterations=1)
            if self.debug_mode:
                self.save_debug_image(dilation, "06_dilated")

            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(dilation, kernel, iterations=1)
            if self.debug_mode:
                self.save_debug_image(erosion, "07_final_preprocessed")

            # return erosion
            return sauvola
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def merge_overlapping_boxes(self, bboxes, y_threshold=5, x_overlap_threshold=5):
        """Merge overlapping bounding boxes with enhanced logic and average threshold-based splitting"""
        merged_bboxes = []
        bboxes = sorted(bboxes, key=lambda x: x[1])  # Sort by y-coordinate

        while bboxes:
            x, y, w, h = bboxes.pop(0)
            merged = False

            for i, (mx, my, mw, mh) in enumerate(merged_bboxes):
                vertical_overlap = (y <= my + mh + y_threshold) and (y + h >= my - y_threshold)
                x_overlap = (x + w >= mx - x_overlap_threshold and x <= mx + mw + x_overlap_threshold)

                if vertical_overlap and x_overlap:
                    new_x = min(mx, x)
                    new_y = min(my, y)
                    new_w = max(mx + mw, x + w) - new_x
                    new_h = max(my + mh, y + h) - new_y
                    merged_bboxes[i] = (new_x, new_y, new_w, new_h)
                    merged = True
                    break

            if not merged:
                merged_bboxes.append((x, y, w, h))

        # Calculate the average width of the bounding boxes
        average_width = sum([w for _, _, w, _ in merged_bboxes]) / len(merged_bboxes)

        # Split boxes that are significantly wider than the average
        refined_bboxes = []
        for x, y, w, h in merged_bboxes:
            if w > 1.5 * average_width:  # Split based on a threshold (2x the average width)
                num_splits = int(w / average_width)  # Determine the number of splits
                for i in range(num_splits):
                    split_x = x + i * (w // num_splits)
                    split_w = w // num_splits
                    refined_bboxes.append((split_x, y, split_w, h))
            else:
                refined_bboxes.append((x, y, w, h))

        return refined_bboxes


    def segment_image(self, image):
        """Enhanced segmentation with better line detection and debug saves"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Edge detection
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(blurred, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            if self.debug_mode:
                self.save_debug_image(edged, "08_edges")

            # Find contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if not cnts:
                raise ValueError("No contours found in image")

            # Sort contours
            (cnts, _) = contours.sort_contours(cnts)

            # Get all bounding boxes
            all_bboxes = []
            for c in cnts:
                if cv2.contourArea(c) < 200:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                
                # Filter out vertically tall boxes based on aspect ratio
                aspect_ratio = h / float(w)
                if aspect_ratio > 5:  # Ignore boxes that are 5 times taller than wide
                    continue
                
                all_bboxes.append((x, y, w, h))

            if not all_bboxes:
                raise ValueError("No valid characters detected")

            # Group into lines
            min_y = min(all_bboxes, key=lambda x: x[1])[1]
            max_y = max(all_bboxes, key=lambda x: x[1])[1]
            avg_height = sum([bbox[3] for bbox in all_bboxes]) / len(all_bboxes)
            num_lines = max(1, int((max_y - min_y) / avg_height))

            # Group boxes by lines with better accuracy
            grouped_lines = {i: [] for i in range(num_lines)}
            for bbox in all_bboxes:
                x, y, w, h = bbox
                line_idx = min(num_lines - 1, int((y - min_y) / ((max_y - min_y) / num_lines)))
                grouped_lines[line_idx].append(bbox)

            # Process each line
            characters = []
            debug_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            for line_idx in sorted(grouped_lines.keys()):
                line = grouped_lines[line_idx]
                if len(line) <= 1:
                    continue

                # Merge overlapping boxes
                merged_line = self.merge_overlapping_boxes(line)
                merged_line = sorted(merged_line, key=lambda x: x[0])  # Sort left to right

                # Extract characters
                line_chars = []
                for x, y, w, h in merged_line:
                    char_img = gray[y:y+h, x:x+w]
                    line_chars.append(char_img)
                    
                    # Draw bounding box for debug
                    if self.debug_mode:
                        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if line_chars:
                    characters.append(line_chars)

            if self.debug_mode:
                self.save_debug_image(debug_image, "09_segmented")

            return characters

        except Exception as e:
            raise ValueError(f"Failed to segment image: {str(e)}")

    def predict_text(self, characters):
        """Text prediction matching the approach from final10.py"""
        if not self.model or not self.char_list:
            raise ValueError("Model not loaded")

        try:
            result = []
            for line_idx, line in enumerate(characters):
                line_predictions = []
                y_prob = []
                for char_idx, char_img in enumerate(line):
                    # Prepare image
                    char_img = cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)
                    img = cv2.resize(char_img, (100, 100))
                    img = img / 255.0

                    # plt.imshow(img)
                    # plt.show()

                    img = np.expand_dims(img, axis=0)
                    images=np.vstack([img])
                    y_prob.append(self.model.predict(images))

                    for i in y_prob:
                        predicted=0
                        predicted=[list(self.train_dataset.class_indices.keys())[i.argmax()]]
                        predicted=predicted[0]
                        
                    predicted=int(predicted)
                    char = self.char_list[predicted]
                    # print(char)

                    line_predictions.append(char)
                    # Debug save if enabled
                    if self.debug_mode:
                        char_debug_dir = self.debug_dir / f"line_{line_idx + 1}"
                        char_debug_dir.mkdir(exist_ok=True)
                        self.save_debug_image(char_img, f"line_{line_idx + 1}/char_{char_idx + 1}")

                    # Predict
                    # pred = self.model.predict(img, verbose=0)
                    # predicted_idx = pred.argmax()

                    # if predicted_idx < len(self.char_list):
                    #     char = self.char_list[predicted_idx]
                    #     line_predictions.append(char)

                # Process line predictions with simple Tamil character handling
                line_text = ""
                temp = ""
                for char in line_predictions:
                    if char in ["ெ", "ை"]:
                        temp = char
                    else:
                        line_text += temp
                        line_text += char
                        temp = ""
                line_text += temp  # Add any remaining modifier
                result.append(line_text)

            return "\n".join(result)

        except Exception as e:
            raise ValueError(f"Failed to predict text: {str(e)}")

    def process_image(self, image_path: str, debug_dir: Path = None) -> dict:
        """Complete pipeline with text prediction"""
        try:
            if debug_dir:
                self.set_debug_mode(True, debug_dir)

            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read image file")

            preprocessed = self.preprocess_image(image)
            characters = self.segment_image(preprocessed)
            if not characters:
                raise ValueError("No characters detected in image")

            # Get prediction with simple Tamil text processing
            predicted_text = self.predict_text(characters)
            
            # Calculate statistics
            stats = {
                "predicted_text": predicted_text,
                "num_lines": len(characters),
                "chars_per_line": [len(line) for line in characters],
                "total_chars": sum(len(line) for line in characters)
            }
            
            return stats

        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
        finally:
            self.debug_mode = False
            self.debug_dir = None

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