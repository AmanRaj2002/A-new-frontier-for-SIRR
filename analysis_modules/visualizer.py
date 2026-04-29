import matplotlib.pyplot as plt
import cv2
import os
import glob
import random

def show_results(dataset_name, input_dir, gt_dir, output_base_dir, num_images=3, randomize=True):
    output_img_dir = os.path.join(output_base_dir, dataset_name, 'Real-img')
    output_map_dir = os.path.join(output_base_dir, dataset_name, 'Real-location_gray')

    if not os.path.exists(output_img_dir):
        print(f"Directory {output_img_dir} does not exist.")
        return

    output_images = sorted(os.listdir(output_img_dir))
    if not output_images:
        print(f"No images found in {output_img_dir}")
        return

    if randomize:
        selected_images = random.sample(output_images, min(num_images, len(output_images)))
    else:
        selected_images = output_images[:num_images]

    fig, axes = plt.subplots(len(selected_images), 4, figsize=(20, 5 * len(selected_images)))
    if len(selected_images) == 1:
        axes = [axes]

    print(f"--- Visualizing {dataset_name} ---")
    for i, out_img_name in enumerate(selected_images):
        base_name = os.path.splitext(out_img_name)[0]
        
        in_match = glob.glob(os.path.join(input_dir, f"{base_name}.*"))
        gt_match = glob.glob(os.path.join(gt_dir, f"{base_name}.*"))
        
        if in_match and gt_match:
            in_img = cv2.cvtColor(cv2.imread(in_match[0]), cv2.COLOR_BGR2RGB)
            gt_img = cv2.cvtColor(cv2.imread(gt_match[0]), cv2.COLOR_BGR2RGB)
            det_map = cv2.imread(os.path.join(output_map_dir, out_img_name), cv2.COLOR_BGR2RGB)
            out_img = cv2.cvtColor(cv2.imread(os.path.join(output_img_dir, out_img_name)), cv2.COLOR_BGR2RGB)
            
            axes[i][0].imshow(in_img)
            if i == 0: axes[i][0].set_title("1. Original Input")
            axes[i][0].set_ylabel(base_name, fontsize=12, rotation=0, labelpad=40)
            axes[i][0].set_xticks([]); axes[i][0].set_yticks([])
            
            axes[i][1].imshow(det_map)
            if i == 0: axes[i][1].set_title("2. Detection Map")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(out_img)
            if i == 0: axes[i][2].set_title("3. Model Output")
            axes[i][2].axis('off')
            
            axes[i][3].imshow(gt_img)
            if i == 0: axes[i][3].set_title("4. Ground Truth")
            axes[i][3].axis('off')
        else:
            print(f"Missing input or GT for {base_name}")
            
    plt.tight_layout()
    plt.show()