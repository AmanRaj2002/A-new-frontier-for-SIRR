import cv2
import matplotlib.pyplot as plt
import os

def show_custom_results(input_path, det_map_path, output_path):
    """Visualizes inference results without Ground Truth."""
    inp = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    det_map = cv2.imread(det_map_path, cv2.IMREAD_GRAYSCALE)
    out = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(inp)
    axes[0].set_title(f"1. Original Input ({os.path.basename(input_path)})")
    axes[0].axis('off')
    
    axes[1].imshow(det_map, cmap='gray')
    axes[1].set_title("2. Model Detection Map")
    axes[1].axis('off')
    
    axes[2].imshow(out)
    axes[2].set_title("3. Network Output")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
# %%
