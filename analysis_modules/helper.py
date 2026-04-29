import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Helper Functions
# ==========================================
def load_rgb_image(path):
    """Safely loads an image and correctly converts it to RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def compute_gradient_magnitude(img):
    """Computes the raw Sobel edge gradient magnitude."""
    # Convert to grayscale for gradient calculation
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients (DO NOT normalize independently!)
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Return raw magnitude for accurate relative comparison
    return cv2.magnitude(grad_x, grad_y)

def get_magnitude_spectrum(img):
    """Computes the centered log-magnitude spectrum of a 2D grayscale image."""
    # Force 2D Grayscale so matplotlib auto-scales properly without clipping
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return np.log(1 + np.abs(fshift))
