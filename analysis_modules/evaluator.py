import os
import glob
import cv2
import torch
import gc
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import matplotlib.image as mpimg

# Suppress fragmentation warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
repo_dir = '/home/amanr.mds2024/Developer/cv_proj1/Reflection_RemoVal_CVPR2024'
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from networks.NAFNet_arch import NAFNet_wDetHead
from networks.network_RefDet import RefDet

class ReflectionEvaluator:
    def __init__(self, repo_dir='/home/amanr.mds2024/Developer/cv_proj1/Reflection_RemoVal_CVPR2024'):
        self.device = self._get_emptiest_gpu()
        print(f"Models assigned to: {self.device}")
        
        self.net_Det = RefDet(backbone='efficientnet-b3', proj_planes=16, pred_planes=32,
                              use_pretrained=False, fix_backbone=False, has_se=False,
                              num_of_layers=6, expansion=4)
        
        self.net_RR = NAFNet_wDetHead(img_channel=3, width=32, middle_blk_num=1,
                                      enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                                      global_residual=False, drop_flag=False, drop_rate=0.4,
                                      concat=True, merge_manner=0)
        
        # Load weights
        det_path = os.path.join(repo_dir, 'ckpt/RD.pth')
        rr_path = os.path.join(repo_dir, 'ckpt/RR.pth')
        
        if not os.path.exists(det_path) or not os.path.exists(rr_path):
            raise FileNotFoundError("Checkpoints missing. Ensure RD.pth and RR.pth are in the ckpt directory.")
            
        self.net_Det.load_state_dict(torch.load(det_path, map_location=self.device))
        self.net_RR.load_state_dict(torch.load(rr_path, map_location=self.device))
        
        self.net_Det.to(self.device).eval()
        self.net_RR.to(self.device).eval()

    def _get_emptiest_gpu(self):
        if not torch.cuda.is_available():
            return torch.device('cpu')
            
        best_gpu, max_free_mem = 0, 0
        for i in range(torch.cuda.device_count()):
            free_mem, _ = torch.cuda.mem_get_info(i)
            print(f"GPU {i} Free Memory: {free_mem / (1024 ** 3):.2f} GB")
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_gpu = i
                
        return torch.device(f'cuda:{best_gpu}')

    def _prep_image(self, img_path, max_dim=1200):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        img_resized = cv2.resize(img, (new_w, new_h))
        
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        return tensor.unsqueeze(0).to(self.device), img_resized

    def _find_gt_match(self, gt_dir, name_no_ext):
        """Resolves naming inconsistencies between blended and GT images."""
        matches = glob.glob(os.path.join(gt_dir, f"{name_no_ext}.*"))
        if matches: return matches[0]
        
        if '-m-' in name_no_ext:
            matches = glob.glob(os.path.join(gt_dir, f"{name_no_ext.replace('-m-', '-g-')}.*"))
        elif name_no_ext.endswith('-m'):
            matches = glob.glob(os.path.join(gt_dir, f"{name_no_ext[:-2] + '-g'}.*"))
            
        return matches[0] if matches else None

    def evaluate_dataset(self, dataset_name, input_dir, gt_dir, output_base_dir):
        output_img_dir = os.path.join(output_base_dir, dataset_name, 'Real-img')
        output_map_dir = os.path.join(output_base_dir, dataset_name, 'Real-location_gray')
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_map_dir, exist_ok=True)

        input_paths = sorted(glob.glob(os.path.join(input_dir, '*.*')))
        psnr_scores, ssim_scores = [], []
        
        print(f"\nEvaluating [{dataset_name}]: {len(input_paths)} images found.")

        with torch.no_grad():
            for in_path in tqdm(input_paths, desc=dataset_name):
                base_name = os.path.basename(in_path)
                name_no_ext = os.path.splitext(base_name)[0]
                
                gt_match = self._find_gt_match(gt_dir, name_no_ext)
                if not gt_match:
                    continue
                    
                in_tensor, _ = self._prep_image(in_path)
                _, gt_img_np = self._prep_image(gt_match)
                
                # Forward Pass
                sparse_out = self.net_Det(in_tensor)
                pred_out = self.net_RR(in_tensor, sparse_out)
                
                # Immediately move to CPU to free VRAM
                pred_np = torch.clamp(pred_out, 0.0, 1.0).squeeze().cpu().numpy().transpose(1, 2, 0)
                map_np = torch.clamp(sparse_out, 0.0, 1.0).squeeze().cpu().numpy()
                gt_np = gt_img_np.astype(np.float32) / 255.0
                
                # Metrics
                psnr_scores.append(calculate_psnr(gt_np, pred_np, data_range=1.0))
                ssim_scores.append(calculate_ssim(gt_np, pred_np, channel_axis=-1, data_range=1.0))
                
                # Save
                mpimg.imsave(os.path.join(output_img_dir, f"{name_no_ext}.png"), pred_np)
                mpimg.imsave(os.path.join(output_map_dir, f"{name_no_ext}.png"), map_np, cmap='gray')

                # Free references; do NOT force gc.collect() inside the loop
                del in_tensor, sparse_out, pred_out

        # Final cleanup post-dataset
        torch.cuda.empty_cache()
        gc.collect()

        return np.mean(psnr_scores) if psnr_scores else 0.0, np.mean(ssim_scores) if ssim_scores else 0.0

    def inference_only(self, dataset_name, input_dir, output_base_dir):
        """Runs the network on custom data without requiring ground truth images."""
        output_img_dir = os.path.join(output_base_dir, dataset_name, 'Real-img')
        output_map_dir = os.path.join(output_base_dir, dataset_name, 'Real-location_gray')
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_map_dir, exist_ok=True)

        input_paths = sorted(glob.glob(os.path.join(input_dir, '*.*')))
        print(f"\nRunning Inference on [{dataset_name}]: {len(input_paths)} custom images found.")

        with torch.no_grad():
            for in_path in tqdm(input_paths, desc=dataset_name):
                base_name = os.path.basename(in_path)
                name_no_ext = os.path.splitext(base_name)[0]
                
                in_tensor, _ = self._prep_image(in_path)
                
                # Forward Pass
                sparse_out = self.net_Det(in_tensor)
                pred_out = self.net_RR(in_tensor, sparse_out)
                
                # Post-process
                pred_np = torch.clamp(pred_out, 0.0, 1.0).squeeze().cpu().numpy().transpose(1, 2, 0)
                map_np = torch.clamp(sparse_out, 0.0, 1.0).squeeze().cpu().numpy()
                
                # Save Outputs
                mpimg.imsave(os.path.join(output_img_dir, f"{name_no_ext}.png"), pred_np)
                mpimg.imsave(os.path.join(output_map_dir, f"{name_no_ext}.png"), map_np, cmap='gray')

                del in_tensor, sparse_out, pred_out

        torch.cuda.empty_cache()
        gc.collect()

    def inference_in_memory(self, input_dir):
        """Runs the network and returns outputs in memory to avoid disk I/O."""
        input_paths = sorted(glob.glob(os.path.join(input_dir, '*.*')))
        results_list = []
        
        print(f"\nRunning In-Memory Inference: {len(input_paths)} images found.")

        with torch.no_grad():
            for in_path in tqdm(input_paths):
                base_name = os.path.basename(in_path)
                
                in_tensor, inp_rgb = self._prep_image(in_path)
                
                # Forward Pass
                sparse_out = self.net_Det(in_tensor)
                pred_out = self.net_RR(in_tensor, sparse_out)
                
                # Convert to numpy arrays immediately
                pred_np = torch.clamp(pred_out, 0.0, 1.0).squeeze().cpu().numpy().transpose(1, 2, 0)
                map_np = torch.clamp(sparse_out, 0.0, 1.0).squeeze().cpu().numpy()
                
                # Store arrays in our list instead of writing to disk
                results_list.append({
                    'filename': base_name,
                    'input_img': inp_rgb,
                    'pred_img': pred_np,
                    'det_map': map_np
                })

                del in_tensor, sparse_out, pred_out

        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return results_list