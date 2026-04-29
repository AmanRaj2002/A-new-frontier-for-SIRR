import glob
import cv2
import time
import os
import pandas as pd

def profile_dataset_resolutions(dataset_base_path, dataset_dict):
    """Scrapes all datasets to build a statistical dataframe of image sizes."""
    stats = []
    
    for name, rel_path in dataset_dict.items():
        input_dir = os.path.join(dataset_base_path, rel_path, 'blended')
        files = glob.glob(os.path.join(input_dir, '*.*'))
        
        if not files:
            continue
            
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                h, w, _ = img.shape
                stats.append({
                    'Dataset': name,
                    'Filename': os.path.basename(f),
                    'Width': w,
                    'Height': h,
                    'Megapixels': round((w * h) / 1000000, 2)
                })
                
    df = pd.DataFrame(stats)
    return df

def generate_summary_report(df):
    """Generates average metrics per dataset."""
    summary = df.groupby('Dataset').agg(
        Total_Images=('Filename', 'count'),
        Avg_Width=('Width', 'mean'),
        Avg_Height=('Height', 'mean'),
        Avg_Megapixels=('Megapixels', 'mean'),
        Max_Megapixels=('Megapixels', 'max')
    ).round(2).reset_index()
    
    return summary