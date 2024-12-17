import streamlit as st
import cv2
from pathlib import Path
import numpy as np

def load_test_images():
    """
    Load test images from the 'pics' directory.

    Returns:
        list: List of image file names.
    """
    script_dir = Path(__file__).resolve().parent
    image_dir = script_dir / "pics"
    
    # Create directory if it doesn't exist
    if not image_dir.exists():
        image_dir.mkdir(parents=True)
        st.warning(f"Created images directory at {image_dir}")
        return []
    
    # List image files with proper error handling
    try:
        image_files = [f.name for f in image_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
        return image_files
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        return []

def safe_read_image(image_path):
    """
    Safely read an image with error handling.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image or None if an error occurred.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        return image
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")
        return None
    
def standardize_image(image):
    """Standardize image size and quality"""
    # Set the standard height
    standard_height = 500  # Set a higher standard height to maintain clarity
    
    # Calculate width while maintaining the original aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    standard_width = int(standard_height * aspect_ratio)
    
    # Resize the image
    resized = cv2.resize(image, (standard_width, standard_height), 
                        interpolation=cv2.INTER_LANCZOS4)  # Use Lanczos interpolation for better quality
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # Adjust contrast and brightness
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_image(image, clip_limit=2.0, tile_grid_size=8):
    """Main function for processing the image"""
    # First, standardize the image
    standardized = standardize_image(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)
    
    # Further image enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Invert the image (to make bands white)
    inverted = cv2.bitwise_not(denoised)
    
    return inverted, standardized