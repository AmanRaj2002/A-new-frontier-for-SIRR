import cv2
import numpy as np
import matplotlib.pyplot as plt
from analysis_modules.helper import load_rgb_image, compute_gradient_magnitude

def analyze_reflection_physics(input_path, gt_path, det_map_path):
    """
    Analyzes the physical properties of reflections vs the model's estimation.
    Uses a 2-row layout to display original images and heatmaps side-by-side.
    
    Data Interpretations per Paper:
    - M_local (JET Heatmap): Binary ground-truth derived from deterministic physical
      gradients (GI > GT). Heatmap shows strict 0 (Deep Blue) or 255 (Bright Red) locations.
    - \hat{M}_local (JET Heatmap): Continuous probability/confidence scores output by the RDNet
     . Smooth gradients from Deep Blue (Low Confidence) to Bright Red
      (High Confidence). Discrepancies between M and \hat{M} are expected due to RDNet's TV loss
      which smooths the prediction and mitigates artifacts present in the noisy mathematical M_local.
    """
    # 1. Load Images
    inp = load_rgb_image(input_path)
    gt = load_rgb_image(gt_path)
    # The det_map is the continuous probability score output \hat{M}_local
    det_map = cv2.imread(det_map_path, cv2.IMREAD_GRAYSCALE)
    
    if det_map is None:
        raise FileNotFoundError(f"Detection map not found at {det_map_path}")
    
    # 2. Ensure dimensions match
    gt = cv2.resize(gt, (inp.shape[1], inp.shape[0]))
    det_map = cv2.resize(det_map, (inp.shape[1], inp.shape[0]))
    
    # 3. Compute Raw Physical Gradients (G_I and G_T) in CV_64F to prevent overflow
    grad_inp = compute_gradient_magnitude(inp)
    grad_gt = compute_gradient_magnitude(gt)
    
    # 4. Implement deterministic MaxRF boundary per paper Eq (1):
    # m_local_true is the physical reflection location GT (binary indicator)
    m_local_true = np.where(grad_inp > grad_gt, 1.0, 0.0).astype(np.float32)
    
    # Normalize the model's output \hat{M}_local probability map to [0, 1]
    m_local_hat = (det_map / 255.0).astype(np.float32)
    
    # 5. Generate Heatmaps using JET
    # True map results in sharp Blue/Red binary heatmap.
    true_heatmap_bgr = cv2.applyColorMap((m_local_true * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # Hat map results in smooth color gradients representing confidence.
    hat_heatmap_bgr = cv2.applyColorMap((m_local_hat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert BGR heatmaps to RGB for Matplotlib
    true_heatmap = cv2.cvtColor(m_local_true, cv2.COLOR_BGR2RGB)
    hat_heatmap = cv2.cvtColor(hat_heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # 6. Visualization: 2-Row Layout to reduce clutter
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Source Images
    axes[0, 0].imshow(inp)
    axes[0, 0].set_title("Input Image ($I$)")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt)
    axes[0, 1].set_title("Transmission Ground Truth ($T$)")
    axes[0, 1].axis('off')
    
    # Row 2: Mathematical GT Locations vs. Model Confidence
    axes[1, 0].imshow(true_heatmap)
    # Binary indicator function GI > GT
    axes[1, 0].set_title("MaxRF Loc GT ($M_{local}$)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hat_heatmap)
    # Continuous confidence score output by Sigmoid layer
    axes[1, 1].set_title("Model Confidence Heatmap ($\hat{M}_{local}$)")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()