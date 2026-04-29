import cv2
import matplotlib.pyplot as plt

def show_in_memory_results(image_data_dict):
    """Visualizes inference results directly from numpy arrays."""
    inp = image_data_dict['input_img']
    out = image_data_dict['pred_img']
    det_map = image_data_dict['det_map']
    filename = image_data_dict['filename']
    
    # Normalize the detection map and apply JET colormap
    heatmap_bgr = cv2.applyColorMap((det_map * 255).astype('uint8'), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].imshow(inp)
    axes[0].set_title(f"1. Original Input (I)\n{filename}")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("2. Estimated Location Map ($\hat{M}_{local}$)")
    axes[1].axis('off')
    
    axes[2].imshow(out)
    axes[2].set_title("3. Network Output ($\hat{T}$)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()