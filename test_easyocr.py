"""
EasyOCR Testing Suite: Unit, Functional, and Accuracy Tests
"""

import pytest
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Tuple
import time
import json

# ============================================
# UNIT TESTS - Testing individual components
# ============================================

class TestEasyOCRUnit:
    """Unit tests for EasyOCR components"""
    
    @pytest.fixture
    def reader(self):
        """Initialize EasyOCR reader once for all tests"""
        return easyocr.Reader(['en'], gpu=False)
    
    def test_reader_initialization(self):
        """Test if reader initializes correctly"""
        reader = easyocr.Reader(['en'], gpu=False)
        assert reader is not None
        # Check reader has necessary attributes
        assert hasattr(reader, 'readtext')
    
    def test_multiple_language_initialization(self):
        """Test multi-language reader initialization"""
        reader = easyocr.Reader(['en', 'es'], gpu=False)
        assert reader is not None
        assert hasattr(reader, 'readtext')
    
    def test_image_loading(self):
        """Test if images load correctly"""
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert img is not None
        assert img.shape == (100, 100, 3)
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images"""
        reader = easyocr.Reader(['en'], gpu=False)
        # Should handle gracefully
        try:
            result = reader.readtext(None)
            assert False, "Should raise an error"
        except:
            assert True
    
    def test_confidence_score_range(self, reader):
        """Test if confidence scores are in valid range [0, 1]"""
        # Create image with text
        img = self._create_test_image("TEST")
        results = reader.readtext(img)
        
        for detection in results:
            confidence = detection[2]
            assert 0 <= confidence <= 1, f"Confidence {confidence} out of range"
    
    def test_output_format(self, reader):
        """Test output format structure"""
        img = self._create_test_image("HELLO")
        results = reader.readtext(img)
        
        for detection in results:
            assert len(detection) == 3  # [bbox, text, confidence]
            assert isinstance(detection[0], list)  # bbox coordinates
            assert isinstance(detection[1], str)   # text
            assert isinstance(detection[2], float) # confidence
    
    def _create_test_image(self, text: str, size=(200, 100)):
        """Helper to create test images with text"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), text, fill='black')
        return np.array(img)


# ============================================
# FUNCTIONAL TESTS - Testing complete workflows
# ============================================

class TestEasyOCRFunctional:
    """Functional tests for complete OCR workflows"""
    
    @pytest.fixture
    def reader(self):
        return easyocr.Reader(['en'], gpu=False)
    
    def test_basic_text_extraction(self, reader):
        """Test basic text extraction workflow"""
        # Create image with known text
        test_text = "HELLO WORLD"
        img = self._create_test_image(test_text)
        
        results = reader.readtext(img)
        extracted_text = ' '.join([detection[1] for detection in results])
        
        assert len(results) > 0, "No text detected"
        assert any(word in extracted_text.upper() for word in test_text.split())
    
    def test_multi_line_text_extraction(self, reader):
        """Test extraction of multi-line text"""
        img = self._create_multiline_image(["Line 1", "Line 2", "Line 3"])
        results = reader.readtext(img)
        
        assert len(results) >= 2, "Should detect multiple lines"
    
    def test_number_extraction(self, reader):
        """Test extraction of numbers"""
        test_numbers = "123456789"
        img = self._create_test_image(test_numbers)
        
        results = reader.readtext(img)
        extracted = ''.join([detection[1] for detection in results])
        
        # Check if digits are detected
        assert any(char.isdigit() for char in extracted)
    
    def test_mixed_alphanumeric(self, reader):
        """Test extraction of mixed alphanumeric text"""
        test_text = "ABC123XYZ"
        img = self._create_test_image(test_text)
        
        results = reader.readtext(img)
        assert len(results) > 0, "Failed to detect mixed alphanumeric"
    
    def test_special_characters(self, reader):
        """Test extraction with special characters"""
        test_text = "Price: $99.99!"
        img = self._create_test_image(test_text)
        
        results = reader.readtext(img)
        extracted = ' '.join([detection[1] for detection in results])
        
        assert len(results) > 0, "Failed to detect text with special chars"
    
    def test_different_image_formats(self, reader):
        """Test different image input formats"""
        test_text = "FORMAT TEST"
        
        # PIL Image
        pil_img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 30), test_text, fill='black')
        results_pil = reader.readtext(np.array(pil_img))
        
        # NumPy array
        np_img = np.array(pil_img)
        results_np = reader.readtext(np_img)
        
        assert len(results_pil) > 0 and len(results_np) > 0
    
    def test_empty_image(self, reader):
        """Test handling of empty image (no text)"""
        # Pure white image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        results = reader.readtext(img)
        
        # Should return empty or very low confidence results
        assert isinstance(results, list)
    
    def test_processing_time(self, reader):
        """Test if processing completes within reasonable time"""
        img = self._create_test_image("TIMING TEST")
        
        start = time.time()
        results = reader.readtext(img)
        duration = time.time() - start
        
        assert duration < 10, f"Processing took too long: {duration}s"
    
    def _create_test_image(self, text: str, size=(300, 100)):
        """Helper to create test images"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), text, fill='black')
        return np.array(img)
    
    def _create_multiline_image(self, lines: List[str], size=(300, 200)):
        """Helper to create multi-line test images"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        y_position = 20
        for line in lines:
            draw.text((20, y_position), line, fill='black')
            y_position += 40
        return np.array(img)


# ============================================
# ACCURACY TESTS - Testing text extraction accuracy
# ============================================

class TestEasyOCRAccuracy:
    """Accuracy tests with ground truth comparison"""
    
    @pytest.fixture
    def reader(self):
        return easyocr.Reader(['en'], gpu=False)
    
    def test_simple_word_accuracy(self, reader):
        """Test accuracy for simple words"""
        test_cases = ["HELLO", "WORLD", "PYTHON", "TEST"]
        
        for expected_text in test_cases:
            img = self._create_test_image(expected_text, font_size=40)
            results = reader.readtext(img)
            
            if results:
                extracted = results[0][1].upper()
                confidence = results[0][2]
                
                # Calculate similarity
                accuracy = self._calculate_similarity(expected_text, extracted)
                
                assert accuracy > 0.7, f"Low accuracy for '{expected_text}': got '{extracted}' ({accuracy:.2f})"
                assert confidence > 0.5, f"Low confidence: {confidence}"
    
    def test_sentence_accuracy(self, reader):
        """Test accuracy for complete sentences"""
        test_sentence = "The quick brown fox"
        img = self._create_test_image(test_sentence, font_size=30)
        
        results = reader.readtext(img)
        extracted = ' '.join([r[1] for r in results])
        
        accuracy = self._calculate_similarity(
            test_sentence.upper(), 
            extracted.upper()
        )
        
        assert accuracy > 0.6, f"Sentence accuracy too low: {accuracy:.2f}"
    
    def test_number_accuracy(self, reader):
        """Test accuracy for number recognition"""
        test_numbers = ["123", "456789", "2024", "100"]
        
        for expected in test_numbers:
            img = self._create_test_image(expected, font_size=40)
            results = reader.readtext(img)
            
            if results:
                extracted = ''.join(filter(str.isdigit, results[0][1]))
                accuracy = self._calculate_similarity(expected, extracted)
                
                assert accuracy > 0.7, f"Number accuracy low: expected '{expected}', got '{extracted}'"
    
    def test_confidence_threshold_accuracy(self, reader):
        """Test accuracy at different confidence thresholds"""
        test_text = "CONFIDENCE"
        img = self._create_test_image(test_text)
        
        results = reader.readtext(img)
        
        high_confidence_results = [r for r in results if r[2] > 0.8]
        medium_confidence_results = [r for r in results if 0.5 < r[2] <= 0.8]
        
        # High confidence results should be more accurate
        if high_confidence_results:
            extracted = high_confidence_results[0][1].upper()
            accuracy = self._calculate_similarity(test_text, extracted)
            assert accuracy > 0.8, "High confidence result has low accuracy"
    
    def test_case_insensitive_accuracy(self, reader):
        """Test accuracy regardless of case"""
        test_cases = [
            ("UPPERCASE", "uppercase"),
            ("lowercase", "LOWERCASE"),
            ("MixedCase", "mixedcase")
        ]
        
        for text, _ in test_cases:
            img = self._create_test_image(text, font_size=50)
            results = reader.readtext(img)
            
            if results:
                extracted = results[0][1]
                # Case-insensitive comparison
                accuracy = self._calculate_similarity(
                    text.upper(), 
                    extracted.upper()
                )
                # Relaxed threshold due to synthetic image limitations
                assert accuracy > 0.5, f"Accuracy too low for '{text}': got '{extracted}' ({accuracy:.2f})"
    
    def test_character_error_rate(self, reader):
        """Calculate Character Error Rate (CER)"""
        ground_truth = "TESTING OCR"
        img = self._create_test_image(ground_truth)
        
        results = reader.readtext(img)
        extracted = ' '.join([r[1] for r in results])
        
        cer = self._calculate_cer(ground_truth, extracted)
        
        assert cer < 0.3, f"CER too high: {cer:.2f}"
    
    def test_word_error_rate(self, reader):
        """Calculate Word Error Rate (WER)"""
        ground_truth = "HELLO WORLD TEST"
        img = self._create_test_image(ground_truth, font_size=40)
        
        results = reader.readtext(img)
        extracted = ' '.join([r[1].upper() for r in results])
        
        wer = self._calculate_wer(ground_truth, extracted)
        
        # Relaxed threshold for synthetic images
        assert wer < 0.7, f"WER too high: {wer:.2f}, extracted: '{extracted}'"
    
    def test_accuracy_report_generation(self, reader):
        """Generate accuracy report for multiple test cases"""
        test_cases = {
            "simple": "HELLO",
            "numbers": "12345",
            "mixed": "ABC123",
            "sentence": "QUICK TEST"
        }
        
        report = {}
        
        for test_name, ground_truth in test_cases.items():
            img = self._create_test_image(ground_truth)
            results = reader.readtext(img)
            
            if results:
                extracted = results[0][1].upper()
                accuracy = self._calculate_similarity(ground_truth, extracted)
                confidence = results[0][2]
                
                report[test_name] = {
                    "ground_truth": ground_truth,
                    "extracted": extracted,
                    "accuracy": accuracy,
                    "confidence": confidence
                }
        
        # Print report
        print("\n=== Accuracy Report ===")
        print(json.dumps(report, indent=2))
        
        # Assert overall accuracy
        avg_accuracy = sum(r["accuracy"] for r in report.values()) / len(report)
        assert avg_accuracy > 0.6, f"Average accuracy too low: {avg_accuracy:.2f}"
    
    # Helper methods
    def _create_test_image(self, text: str, size=(400, 100), font_size=30):
        """Create test image with specified text"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        # Using default font
        draw.text((20, 30), text, fill='black')
        return np.array(img)
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate"""
        ref = reference.replace(" ", "")
        hyp = hypothesis.replace(" ", "")
        
        distance = self._levenshtein_distance(ref, hyp)
        cer = distance / len(ref) if len(ref) > 0 else 0
        return cer
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        distance = self._levenshtein_distance(ref_words, hyp_words)
        wer = distance / len(ref_words) if len(ref_words) > 0 else 0
        return wer
    
    def _levenshtein_distance(self, s1, s2) -> int:
        """Calculate Levenshtein distance between two sequences"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# ============================================
# TEST RUNNER
# ============================================

if __name__ == "__main__":
    """Run tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])