"""
Alternative text_recognizer.py using EasyOCR (more stable on Mac)
Save as src/recognition/text_recognizer.py if PaddleOCR has issues
"""

import cv2
import numpy as np
import easyocr
import re

class TextRecognizer:
    def __init__(self, lang='en'):
        """Initialize EasyOCR with specified language"""
        print("Initializing EasyOCR (this may take a moment on first run)...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR initialized successfully!")
    
    def recognize_text(self, image, preprocess=True):
        """
        Recognize text in image and return structured results
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Simple preprocessing
        if preprocess:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # EasyOCR detection
        result = self.reader.readtext(image)
        
        if not result:
            return []
        
        text_data = []
        for detection in result:
            bbox = detection[0]
            text = detection[1]
            confidence = detection[2]
            
            # Convert bbox format
            bbox_array = np.array(bbox)
            x_min, y_min = bbox_array[:, 0].min(), bbox_array[:, 1].min()
            x_max, y_max = bbox_array[:, 0].max(), bbox_array[:, 1].max()
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            text_data.append({
                'bbox': bbox,
                'text': text.strip(),
                'confidence': confidence,
                'center': (center_x, center_y),
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'width': width,
                'height': height
            })
        
        return text_data
    
    def classify_text(self, text_data):
        """
        Classify text into labels, values, and symbols
        """
        labels = []
        values = []
        symbols = []
        
        value_pattern = r'^[\d\.\+\-\±\∅\⌀Ø°x\|]+$'
        label_keywords = [
            'thickness', 'wall', 'length', 'chamfer', 'diameter', 'dia',
            'perpendicularity', 'concentricity', 'step', 'hole', 'drill',
            'major', 'minor', 'total', 'id', 'od', 'cd', 'c.d', 'he', 'a',
            'width', 'height', 'depth'
        ]
        
        for item in text_data:
            text = item['text'].lower()
            original_text = item['text']
            
            # Classify as symbol
            if len(original_text) <= 2 and any(c in original_text for c in ['∅', '⌀', 'Ø', '°', '±']):
                item['type'] = 'symbol'
                symbols.append(item)
            # Classify as label
            elif any(keyword in text for keyword in label_keywords):
                item['type'] = 'label'
                labels.append(item)
            # Classify as value
            elif re.search(value_pattern, original_text.replace(' ', '')):
                item['type'] = 'value'
                values.append(item)
            # Handle colon-separated
            elif ':' in original_text:
                parts = original_text.split(':')
                if len(parts) == 2:
                    item['type'] = 'label'
                    labels.append(item)
                    # Create value item
                    value_item = item.copy()
                    value_item['text'] = parts[1].strip()
                    value_item['type'] = 'value'
                    if value_item['text']:
                        values.append(value_item)
            else:
                # Default classification based on length
                if len(original_text) > 2:
                    item['type'] = 'label'
                    labels.append(item)
                else:
                    item['type'] = 'value'
                    values.append(item)
        
        return labels, values, symbols
    
    def pair_label_value(self, labels, values, max_distance=200):
        """
        Pair labels with their corresponding values based on proximity
        """
        pairs = []
        used_values = set()
        
        for label in labels:
            best_match = None
            min_distance = float('inf')
            
            label_center = label['center']
            
            for idx, value in enumerate(values):
                if idx in used_values:
                    continue
                
                value_center = value['center']
                
                # Calculate Euclidean distance
                distance = np.sqrt(
                    (label_center[0] - value_center[0])**2 + 
                    (label_center[1] - value_center[1])**2
                )
                
                # Calculate direction
                dx = value_center[0] - label_center[0]
                dy = value_center[1] - label_center[1]
                
                # Apply penalties for wrong directions
                if dx < -20 or dy < -50:
                    distance *= 2
                
                # Apply bonus for horizontal alignment
                if abs(dy) < 30:
                    distance *= 0.5
                
                # Find best match
                if distance < min_distance and distance < max_distance:
                    min_distance = distance
                    best_match = idx
            
            if best_match is not None:
                pairs.append({
                    'label': label['text'],
                    'value': values[best_match]['text'],
                    'label_bbox': label['bbox'],
                    'value_bbox': values[best_match]['bbox'],
                    'confidence': (label['confidence'] + values[best_match]['confidence']) / 2
                })
                used_values.add(best_match)
            else:
                # Label without value
                pairs.append({
                    'label': label['text'],
                    'value': None,
                    'label_bbox': label['bbox'],
                    'value_bbox': None,
                    'confidence': label['confidence']
                })
        
        return pairs
    
    def extract_dimensions(self, image):
        """
        Complete pipeline: recognize, classify, and pair text
        """
        # Step 1: Recognize text
        text_data = self.recognize_text(image)
        
        if not text_data:
            return {
                'text_data': [],
                'labels': [],
                'values': [],
                'symbols': [],
                'pairs': []
            }
        
        # Step 2: Classify text
        labels, values, symbols = self.classify_text(text_data)
        
        # Step 3: Pair labels with values
        pairs = self.pair_label_value(labels, values)
        
        return {
            'text_data': text_data,
            'labels': labels,
            'values': values,
            'symbols': symbols,
            'pairs': pairs
        }
    
    def visualize_results(self, image, results, output_path=None):
        """
        Visualize detection results with bounding boxes
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        vis_image = image.copy()
        
        # Draw all text with bounding boxes
        for item in results['text_data']:
            bbox = np.array(item['bbox'], dtype=np.int32)
            text_type = item.get('type', 'unknown')
            
            # Color coding
            if text_type == 'label':
                color = (255, 0, 0)  # Blue
            elif text_type == 'value':
                color = (0, 255, 0)  # Green
            elif text_type == 'symbol':
                color = (0, 0, 255)  # Red
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw bounding box
            cv2.polylines(vis_image, [bbox], True, color, 2)
            
            # Put text label
            cv2.putText(
                vis_image,
                f"{item['text']} ({text_type})",
                (int(item['x_min']), int(item['y_min']) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        # Draw connections between paired labels and values
        for pair in results['pairs']:
            if pair['value'] is not None:
                label_bbox = np.array(pair['label_bbox'])
                value_bbox = np.array(pair['value_bbox'])
                
                # Calculate centers
                label_center = label_bbox.mean(axis=0).astype(int)
                value_center = value_bbox.mean(axis=0).astype(int)
                
                # Draw line
                cv2.line(vis_image, tuple(label_center), tuple(value_center), (255, 255, 0), 1)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def export_to_dict(self, pairs):
        """
        Export pairs to a simple dictionary format
        """
        result = {}
        for pair in pairs:
            label = pair['label']
            value = pair['value'] if pair['value'] else ""
            result[label] = value
        
        return result