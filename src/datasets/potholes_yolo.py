import cv2
import torch
import numpy as np
import lightning as L
from torch.utils.data import DataLoader
import kagglehub
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data_utils import parse_xmls 

# ==========================================
# 1. DATASET (Przetwarzanie pojedynczych obrazów)
# ==========================================
class PotholeDatasetYolo(torch.utils.data.Dataset):
    def __init__(self, df, img_size: int = 640, is_train: bool = True):
        self.df = df
        self.images = df['file'].unique()
        self.img_size = img_size
        self.is_train = is_train

        # NAPRAWA WARNINGA: Zmiana `value=0` na `fill=0` i `border_mode=0`
        if self.is_train:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
                A.HorizontalFlip(p=0.5), 
                A.RandomBrightnessContrast(p=0.5), 
                A.MotionBlur(p=0.2), 
                A.GaussNoise(p=0.2), 
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=0),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            orig_h, orig_w = self.img_size, self.img_size
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]

        img_data = self.df[self.df["file"] == img_path]
        
        bboxes = []
        labels = []
        for _, row in img_data.iterrows():
            xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])

            # Obcinanie ramek wychodzących poza ekran
            xmin = max(0.0, min(float(orig_w), xmin))
            ymin = max(0.0, min(float(orig_h), ymin))
            xmax = max(0.0, min(float(orig_w), xmax))
            ymax = max(0.0, min(float(orig_h), ymax))

            if xmax > xmin and ymax > ymin:
                bboxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(row['label']))

        if len(bboxes) == 0:
            transformed = self.transform(image=img, bboxes=[], class_labels=[])
        else:
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)
            
        img_tensor = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']

        yolo_targets = []
        for bbox, cls_id in zip(transformed_bboxes, transformed_labels):
            xmin, ymin, xmax, ymax = bbox
            x_center = ((xmin + xmax) / 2.0) / self.img_size
            y_center = ((ymin + ymax) / 2.0) / self.img_size
            w = (xmax - xmin) / self.img_size
            h = (ymax - ymin) / self.img_size
            yolo_targets.append([cls_id, x_center, y_center, w, h])

        if len(yolo_targets) > 0:
            target_tensor = torch.as_tensor(yolo_targets, dtype=torch.float32)
        else:
            target_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, target_tensor


# ==========================================
# 2. DATA MODULE (Integracja z PyTorch Lightning i Fiddle)
# ==========================================
class PotholeDataModuleYOLO(L.LightningDataModule):
    def __init__(self, batch_size: int = 16, img_size: int = 640):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_path = None

    def prepare_data(self):
        self.data_path = kagglehub.dataset_download("idanbaru/annotated-potholes-with-severity-levels")

    def setup(self, stage=None):
        df = parse_xmls(self.data_path)        
        
        images = df['file'].unique()
        split_idx = int(len(images) * 0.8)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        df_train = df[df['file'].isin(train_images)]
        df_val = df[df['file'].isin(val_images)]
        
        self.train_ds = PotholeDatasetYolo(df_train, img_size=self.img_size, is_train=True)
        self.val_ds = PotholeDatasetYolo(df_val, img_size=self.img_size, is_train=False)

    def collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        
        new_targets = []
        for i, target in enumerate(targets):
            if target.shape[0] > 0:
                img_idx = torch.full((target.shape[0], 1), i, dtype=torch.float32)
                new_targets.append(torch.cat([img_idx, target], dim=1))
        
        if len(new_targets) > 0:
            targets_combined = torch.cat(new_targets, dim=0)
        else:
            targets_combined = torch.zeros((0, 6))

        return images, {
            "batch_idx": targets_combined[:, 0],
            "cls": targets_combined[:, 1],
            "bboxes": targets_combined[:, 2:],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )