import cv2
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.helper import load_rgb_image, compute_gradient_magnitude, get_magnitude_spectrum

# def get_magnitude_spectrum(img_gray):
#     """Computes the centered log-magnitude spectrum of an image."""
#     f = np.fft.fft2(img_gray)
#     fshift = np.fft.fftshift(f)
#     return np.log(1 + np.abs(fshift))

def analyze_amplified_differences(input_path, gt_path, output_path, scale_factor=5.0):
    """Visualizes the scaled residuals for both Spatial and Frequency domains."""
    inp = load_rgb_image(input_path).astype(np.float32)
    gt = load_rgb_image(gt_path).astype(np.float32)
    out = load_rgb_image(output_path).astype(np.float32)
    
    gt = cv2.resize(gt, (inp.shape[1], inp.shape[0]))
    out = cv2.resize(out, (inp.shape[1], inp.shape[0]))
    
    # Spatial differences
    true_reflection_spatial = np.abs(inp - gt)
    model_error_spatial = np.abs(out - gt)
    
    # Flatten to 2D by taking the mean across RGB channels to map to 'turbo' colormap cleanly
    true_ref_vis = np.clip(np.mean(true_reflection_spatial, axis=-1) * scale_factor, 0, 255).astype(np.uint8)
    model_error_vis = np.clip(np.mean(model_error_spatial, axis=-1) * scale_factor, 0, 255).astype(np.uint8)
    
    # Frequency differences
    mag_inp = get_magnitude_spectrum(inp)
    mag_gt = get_magnitude_spectrum(gt)
    mag_out = get_magnitude_spectrum(out)
    
    true_ref_freq = np.abs(mag_inp - mag_gt)
    model_error_freq = np.abs(mag_out - mag_gt)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    axes[0, 0].imshow(true_ref_vis, cmap='turbo'); axes[0, 0].set_title(f"True Spatial Reflection (Amplified {scale_factor}x)\n$|I_{{in}} - I_{{gt}}|$"); axes[0, 0].axis('off')
    axes[0, 1].imshow(model_error_vis, cmap='turbo'); axes[0, 1].set_title(f"Model Spatial Error (Amplified {scale_factor}x)\n$|I_{{out}} - I_{{gt}}|$"); axes[0, 1].axis('off')
    
    axes[1, 0].imshow(true_ref_freq, cmap='turbo'); axes[1, 0].set_title("True Frequency Reflection Error\n$|S_{{in}} - S_{{gt}}|$"); axes[1, 0].axis('off')
    axes[1, 1].imshow(model_error_freq, cmap='turbo'); axes[1, 1].set_title("Model Frequency Residual Error\n$|S_{{out}} - S_{{gt}}|$"); axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()