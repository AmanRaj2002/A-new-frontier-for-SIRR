import cv2
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.helper import load_rgb_image, get_magnitude_spectrum

def analyze_frequency_domain(input_path, gt_path, output_path):
    """
    Visualizes the spatial and frequency domains side-by-side to analyze
    how well the neural network removes reflection frequencies.
    """
    inp = load_rgb_image(input_path)
    gt = load_rgb_image(gt_path)
    out = load_rgb_image(output_path)
    
    # Ensure dimensions match
    gt = cv2.resize(gt, (inp.shape[1], inp.shape[0]))
    out = cv2.resize(out, (inp.shape[1], inp.shape[0]))
    
    # Compute Magnitude Spectra
    mag_inp = get_magnitude_spectrum(inp)
    mag_gt = get_magnitude_spectrum(gt)
    mag_out = get_magnitude_spectrum(out)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Spatial Domain
    axes[0, 0].imshow(inp)
    axes[0, 0].set_title("Input (Spatial)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(out)
    axes[0, 1].set_title("Network Output (Spatial)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt)
    axes[0, 2].set_title("Ground Truth (Spatial)")
    axes[0, 2].axis('off')
    
    # Row 2: Frequency Domain (Magnitude Spectrum)
    axes[1, 0].imshow(mag_inp, cmap='magma')
    axes[1, 0].set_title("Input (Magnitude Spectrum)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mag_out, cmap='magma')
    axes[1, 1].set_title("Output (Magnitude Spectrum)")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mag_gt, cmap='magma')
    axes[1, 2].set_title("GT (Magnitude Spectrum)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()