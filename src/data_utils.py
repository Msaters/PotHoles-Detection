import cv2
import os
import torch
import numpy as np
import kagglehub
import xml.etree.ElementTree as ET
import glob
import pandas as pd

PATH = kagglehub.dataset_download("idanbaru/annotated-potholes-with-severity-levels")

LABEL_MAP = {
    "minor_pothole": 1,
    "medium_pothole": 2,
    "major_pothole": 3
}

def parse_xmls(path):

    df = []

    img_dir = os.path.join(path, 'images')
    annots_path = os.path.join(path, 'annotations/*.xml')

    for xml_file in glob.glob(annots_path): # find pathnames that matches a pattern
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text
        img_path = os.path.join(img_dir, filename)

        for obj in root.findall("object"):

            severity = obj.find('name').text.lower()

            df.append({
                "file": img_path,
                "xmin": int(obj.find('bndbox/xmin').text),
                "ymin": int(obj.find('bndbox/ymin').text),
                "xmax": int(obj.find('bndbox/xmax').text),
                "ymax": int(obj.find('bndbox/ymax').text),
                "label": LABEL_MAP.get(severity, 1)
            })

    return pd.DataFrame(df)

class PotholeDataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.df = df
        self.images = df['file'].unique()
    
    def __getitem__(self, idx):

        img_path = self.images[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_data = self.df[self.df["file"] == img_path]
        
        boxes = img_data[["xmin", "ymin", "xmax", "ymax"]].values
        labels = img_data["label"].values

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        # TODO: add augmentations

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, target

    def __len__(self):
        return len(self.images)


class PatchPotholeDataset(torch.utils.data.Dataset):
    """Grid-based pothole dataset for patch-level detection.
    
    Divides images into an NxN grid and creates binary targets for each grid cell,
    marking cells that contain pothole bounding boxes.
    
    Args:
        df: DataFrame with columns ['file', 'xmin', 'ymin', 'xmax', 'ymax']
        img_size: Target image size (default: 256)
        grid_size: Grid dimension (default: 8 for 8x8 grid)
    """
    
    def __init__(self, df: pd.DataFrame, img_size: int = 256, grid_size: int = 8):
        self.df = df
        self.images = df['file'].unique()
        self.img_size = img_size
        self.grid_size = grid_size
        self.cell_size = img_size / grid_size

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        
        # Load and convert image
        img = cv2.imread(img_path)
        
        # Handle corrupted/missing images
        if img is None:
            # Return black image as fallback
            print(f"⚠️  Warning: Cannot read image {img_path}, using black image")
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        
        # Resize image to fixed dimensions
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize and convert to PyTorch tensor (C, H, W)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Create empty grid target (1, grid_size, grid_size) filled with zeros
        target_grid = torch.zeros((1, self.grid_size, self.grid_size), dtype=torch.float32)
        
        # Get all annotations for this image
        img_data = self.df[self.df["file"] == img_path]
        
        # Mark grid cells that contain bounding boxes
        for _, row in img_data.iterrows():
            # Scale bounding box coordinates to resized image
            xmin = int(row['xmin'] * (self.img_size / orig_w))
            ymin = int(row['ymin'] * (self.img_size / orig_h))
            xmax = int(row['xmax'] * (self.img_size / orig_w))
            ymax = int(row['ymax'] * (self.img_size / orig_h))
            
            # Calculate which grid cells intersect with the bounding box
            grid_xmin = max(0, int(xmin // self.cell_size))
            grid_ymin = max(0, int(ymin // self.cell_size))
            grid_xmax = min(self.grid_size - 1, int(xmax // self.cell_size))
            grid_ymax = min(self.grid_size - 1, int(ymax // self.cell_size))
            
            # Set 1 in corresponding grid cells
            target_grid[0, grid_ymin:grid_ymax+1, grid_xmin:grid_xmax+1] = 1.0

        return img_tensor, target_grid

    def __len__(self) -> int:
        return len(self.images)