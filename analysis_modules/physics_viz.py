import cv2
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.helper import load_rgb_image, compute_gradient_magnitude, get_magnitude_spectrum

# def compute_gradient_magnitude(img):
#     """Computes the Sobel edge gradient magnitude for grayscale or color images."""
    
#     # Convert to grayscale if input is color
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # The paper mentions applying the Sobel operator to compute gradient maps
#     grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
#     magnitude = cv2.magnitude(grad_x, grad_y)
    
#     # Normalize to 0–255
#     magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
#     return magnitude.astype(np.uint8)

def analyze_reflection_physics(input_path, gt_path, det_map_path):
    """Visualizes the MaxRF indicator function vs the model's detection map."""
    inp = load_rgb_image(input_path)
    gt = load_rgb_image(gt_path)
    
    # Load detection map explicitly in GRAYSCALE
    det_map = cv2.imread(det_map_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure dimensions match
    gt = cv2.resize(gt, (inp.shape[1], inp.shape[0]))
    det_map = cv2.resize(det_map, (inp.shape[1], inp.shape[0]))
    
    # Compute Raw Physical Gradients (G_I and G_T)
    grad_inp = compute_gradient_magnitude(inp)
    grad_gt = compute_gradient_magnitude(gt)
    
    # Implement the Maximum Reflection Filter (MaxRF) as defined in Eq. 1
    # M_local = 1 if G_I > G_T else 0
    epsilon = 0.0 
    
    # M_local = 1 if G_I > (G_T + epsilon) else 0
    m_local_true = np.where(grad_inp > (grad_gt + epsilon), 1.0, 0.0).astype(np.float32)
    
    # Normalize the model's detection map to [0, 1]
    m_local_hat = (det_map / 255.0).astype(np.float32)
    
    # Calculate Error Map
    error_map = np.abs(m_local_true - m_local_hat)
    
    # Convert for heatmap colormap (0-255 scale)
    error_map_vis = (error_map * 255).astype(np.uint8)
    error_heatmap = cv2.applyColorMap(error_map_vis, cv2.COLORMAP_JET)
    
    # Normalize input gradient just for the plot display
    grad_inp_vis = cv2.normalize(grad_inp, None, 0, 1, cv2.NORM_MINMAX)
    grad_gt_vis = cv2.normalize(grad_gt, None, 0, 1, cv2.NORM_MINMAX)
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    axes[0].imshow(grad_inp_vis, cmap='gray'); axes[0].set_title("Input Gradient ($G_I$)"); axes[0].axis('off')
    axes[1].imshow(grad_gt_vis, cmap='gray'); axes[1].set_title("Transmission Gradient ($G_T$)"); axes[1].axis('off')
    axes[2].imshow(m_local_true, cmap='gray'); axes[2].set_title("MaxRF Indicator ($M_{local}$)"); axes[2].axis('off')
    axes[3].imshow(m_local_hat, cmap='gray'); axes[3].set_title("Model's Estimate ($\\hat{M}_{local}$)"); axes[3].axis('off')
    # axes[3].imshow(cv2.cvtColor(error_heatmap, cv2.COLOR_BGR2RGB)); axes[3].set_title("Detection Error Heatmap"); axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()