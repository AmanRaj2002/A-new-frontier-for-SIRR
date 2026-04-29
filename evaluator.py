import os
import sys
import glob
import cv2
import torch
import gc
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import matplotlib.image as mpimg

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

repo_dir = './Reflection_RemoVal_CVPR2024'
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from networks.NAFNet_arch import NAFNet_wDetHead
from networks.network_RefDet import RefDet

class ReflectionEvaluator:
    def __init__(self, repo_dir='./Reflection_RemoVal_CVPR2024'):
        self.device = self._get_emptiest_gpu()
        print(f"Models assigned to: {self.device}")
        
        self.net_Det = RefDet(backbone='efficientnet-b3', proj_planes=16, pred_planes=32,
                              use_pretrained=False, fix_backbone=False, has_se=False,
                              num_of_layers=6, expansion=4)
        
        self.net_RR = NAFNet_wDetHead(img_channel=3, width=32, middle_blk_num=1,
                                      enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                                      global_residual=False, drop_flag=False, drop_rate=0.4,
                                      concat=True, merge_manner=0)
        
        self.net_Det.load_state_dict(torch.load(os.path.join(repo_dir, 'ckpt/RD.pth'), map_location=self.device))
        self.net_RR.load_state_dict(torch.load(os.path.join(repo_dir, 'ckpt/RR.pth'), map_location=self.device))
        
        self.net_Det.to(self.device).eval()
        self.net_RR.to(self.device).eval()

    def _get_emptiest_gpu(self):
        if not torch.cuda.is_available():
            return torch.device('cpu')
            
        best_gpu = 0
        max_free_mem = 0
        
        for i in range(torch.cuda.device_count()):
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024 ** 3)
            print(f"GPU {i} Free Memory: {free_gb:.2f} GB")
            
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
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]

        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        img_resized = cv2.resize(img, (new_w, new_h))
        
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        return tensor.unsqueeze(0).to(self.device), img_resized

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
                
                # --- SMART GT MATCHING ---
                # 1. Try exact match first
                gt_matches = glob.glob(os.path.join(gt_dir, f"{name_no_ext}.*"))
                
                # 2. Try SIR2 format (-m- in the middle)
                if not gt_matches and '-m-' in name_no_ext:
                    sir2_name = name_no_ext.replace('-m-', '-g-')
                    gt_matches = glob.glob(os.path.join(gt_dir, f"{sir2_name}.*"))
                
                # 3. Try SIR2 format (-m at the very end)
                if not gt_matches and name_no_ext.endswith('-m'):
                    sir2_name = name_no_ext[:-2] + '-g'
                    gt_matches = glob.glob(os.path.join(gt_dir, f"{sir2_name}.*"))
                
                # If still no match, skip
                if not gt_matches:
                    continue
                    
                in_tensor, _ = self._prep_image(in_path)
                gt_tensor, gt_img_np = self._prep_image(gt_matches[0])
                
                # Forward Pass
                sparse_out = self.net_Det(in_tensor)
                pred_out = self.net_RR(in_tensor, sparse_out)
                
                # Post-process
                pred_np = torch.clamp(pred_out, 0.0, 1.0).squeeze().cpu().numpy().transpose(1, 2, 0)
                map_np = torch.clamp(sparse_out, 0.0, 1.0).squeeze().cpu().numpy()
                gt_np = gt_img_np.astype(np.float32) / 255.0
                
                # Metrics
                psnr_scores.append(calculate_psnr(gt_np, pred_np, data_range=1.0))
                ssim_scores.append(calculate_ssim(gt_np, pred_np, channel_axis=-1, data_range=1.0))
                
                # Save
                mpimg.imsave(os.path.join(output_img_dir, f"{name_no_ext}.png"), pred_np)
                mpimg.imsave(os.path.join(output_map_dir, f"{name_no_ext}.png"), map_np, cmap='gray')

                # Aggressive Cleanup
                del in_tensor, gt_tensor, sparse_out, pred_out
                torch.cuda.empty_cache()
                gc.collect()

        return np.mean(psnr_scores) if psnr_scores else 0.0, np.mean(ssim_scores) if ssim_scores else 0.0
    
    def inference_only(self, dataset_name, input_dir, output_base_dir):
        """Runs the network without requiring ground truth images."""
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

                # Aggressive Cleanup
                del in_tensor, sparse_out, pred_out
                torch.cuda.empty_cache()
                gc.collect()