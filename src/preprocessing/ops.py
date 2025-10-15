import cv2
import numpy as np

def preprocess_engineering_drawing(image, target_size=None):
    """
    Preprocess engineering drawing for better OCR results
    
    Args:
        image: Input image (BGR format)
        target_size: Tuple (width, height) to resize image, or None to keep original
    
    Returns:
        Preprocessed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Morphological operations to enhance text
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to BGR for OCR
    result = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Resize if needed
    if target_size:
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_CUBIC)
    
    return result

def enhance_contrast(image):
    """
    Enhance contrast using CLAHE
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(image)
    
    return result

def remove_noise(image):
    """
    Remove noise from image
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Apply Non-local Means Denoising
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    return denoised

def deskew_image(image):
    """
    Deskew image if it's rotated
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Get coordinates of all white pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    # Calculate rotation angle
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Only deskew if angle is significant
    if abs(angle) > 0.5:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated
    
    return image

def pipeline_preprocess(image, enhance=True, denoise=True, deskew=False):
    """
    Complete preprocessing pipeline
    
    Args:
        image: Input image
        enhance: Whether to enhance contrast
        denoise: Whether to remove noise
        deskew: Whether to deskew image
    
    Returns:
        Preprocessed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    result = image.copy()
    
    if deskew:
        result = deskew_image(result)
    
    if denoise:
        result = remove_noise(result)
    
    if enhance:
        result = enhance_contrast(result)
    
    return result