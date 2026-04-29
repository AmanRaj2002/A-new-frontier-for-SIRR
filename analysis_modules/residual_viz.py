import cv2
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.helper import load_rgb_image, get_magnitude_spectrum

def analyze_amplified_differences(input_path, gt_path, output_path, scale_factor=5.0):
    """
    Visualizes the scaled residuals for both Spatial and Frequency domains.
    Amplifies the errors so human eyes can easily detect structural remnants.
    """
    # Load as float32 to prevent underflow/overflow during subtraction
    inp = load_rgb_image(input_path).astype(np.float32)
    gt = load_rgb_image(gt_path).astype(np.float32)
    out = load_rgb_image(output_path).astype(np.float32)
    
    gt = cv2.resize(gt, (inp.shape[1], inp.shape[0]))
    out = cv2.resize(out, (inp.shape[1], inp.shape[0]))
    
    # Calculate Spatial differences
    true_reflection_spatial = np.abs(inp - gt)
    model_error_spatial = np.abs(out - gt)
    
    # Flatten to 2D by taking the mean across RGB channels to map cleanly to 'turbo' colormap
    true_ref_vis = np.clip(np.mean(true_reflection_spatial, axis=-1) * scale_factor, 0, 255).astype(np.uint8)
    model_error_vis = np.clip(np.mean(model_error_spatial, axis=-1) * scale_factor, 0, 255).astype(np.uint8)
    
    # Calculate Frequency differences (Requires uint8 format)
    mag_inp = get_magnitude_spectrum(inp.astype(np.uint8))
    mag_gt = get_magnitude_spectrum(gt.astype(np.uint8))
    mag_out = get_magnitude_spectrum(out.astype(np.uint8))
    
    true_ref_freq = np.abs(mag_inp - mag_gt)
    model_error_freq = np.abs(mag_out - mag_gt)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].imshow(true_ref_vis, cmap='turbo')
    axes[0, 0].set_title(f"True Spatial Reflection (Amplified {scale_factor}x)\n$|I - T|$")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(model_error_vis, cmap='turbo')
    axes[0, 1].set_title(f"Model Spatial Error (Amplified {scale_factor}x)\n$|\\hat{{T}} - T|$")
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(true_ref_freq, cmap='turbo')
    axes[1, 0].set_title("True Frequency Reflection Error\n$|S_I - S_T|$")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(model_error_freq, cmap='turbo')
    axes[1, 1].set_title("Model Frequency Residual Error\n$|S_{\\hat{T}} - S_T|$")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()