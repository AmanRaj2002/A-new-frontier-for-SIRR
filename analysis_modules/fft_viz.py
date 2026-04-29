import cv2
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.helper import load_rgb_image, compute_gradient_magnitude, get_magnitude_spectrum

# def get_magnitude_spectrum(img_gray):
#     """Computes the centered log-magnitude spectrum of an image."""
#     # 1. Compute 2D FFT
#     f = np.fft.fft2(img_gray)
#     # 2. Shift the zero-frequency component to the center of the spectrum
#     fshift = np.fft.fftshift(f)
#     # 3. Calculate magnitude and apply log scale
#     # Add 1 to avoid log(0)
#     magnitude_spectrum = np.log(1 + np.abs(fshift))
#     return magnitude_spectrum

def analyze_frequency_domain(input_path, gt_path, output_path):
    """Visualizes the spatial and frequency domains side-by-side."""
    inp = load_rgb_image(input_path)
    gt = load_rgb_image(gt_path)
    out = load_rgb_image(output_path)
    
    gt = cv2.resize(gt, (inp.shape[1], inp.shape[0]))
    out = cv2.resize(out, (inp.shape[1], inp.shape[0]))
    
    # Compute Spectra (using 2D grayscale internally to avoid imshow clipping)
    mag_inp = get_magnitude_spectrum(inp)
    mag_gt = get_magnitude_spectrum(gt)
    mag_out = get_magnitude_spectrum(out)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].imshow(inp); axes[0, 0].set_title("Input (Spatial)"); axes[0, 0].axis('off')
    axes[0, 1].imshow(out); axes[0, 1].set_title("Network Output (Spatial)"); axes[0, 1].axis('off')
    axes[0, 2].imshow(gt); axes[0, 2].set_title("Ground Truth (Spatial)"); axes[0, 2].axis('off')
    
    axes[1, 0].imshow(mag_inp, cmap='magma'); axes[1, 0].set_title("Input (Magnitude Spectrum)"); axes[1, 0].axis('off')
    axes[1, 1].imshow(mag_out, cmap='magma'); axes[1, 1].set_title("Output (Magnitude Spectrum)"); axes[1, 1].axis('off')
    axes[1, 2].imshow(mag_gt, cmap='magma'); axes[1, 2].set_title("GT (Magnitude Spectrum)"); axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()