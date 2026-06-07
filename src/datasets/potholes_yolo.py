import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PotholeDatasetYolo(torch.utils.data.Dataset):
    def __init__(self, df, img_size: int = 640, is_train: bool = True):
        self.df = df
        self.images = df['file'].unique()
        self.img_size = img_size
        self.is_train = is_train

        # 1. Definiujemy transformacje dla zbioru TRENINGOWEGO
        if self.is_train:
            self.transform = A.Compose([
                # Zachowanie proporcji obrazu (tzw. Letterboxing)
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                
                # --- WŁAŚCIWE AUGMENTACJE ---
                A.HorizontalFlip(p=0.5), # Szansa 50% na odbicie lustrzane drogi
                A.RandomBrightnessContrast(p=0.5), # Zmiany oświetlenia (pochmurno/słonecznie)
                A.MotionBlur(p=0.2), # Imitacja rozmazania od pędu kamery samochodowej
                A.GaussNoise(p=0.2), # Szum matrycy z taniej kamery
                
                # Przygotowanie dla PyTorcha (skalowanie do 0-1)
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
        # 2. Definiujemy transformacje dla zbioru WALIDACYJNEGO (bez przekształceń!)
        else:
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        # Wczytanie obrazu
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_data = self.df[self.df["file"] == img_path]
        
        # Wyciągamy boxy w absolutnych pikselach (pascal_voc: xmin, ymin, xmax, ymax)
        bboxes = []
        labels = []
        for _, row in img_data.iterrows():
            # Zabezpieczenie przed ułamkowymi pikselami z XML
            bboxes.append([float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])])
            labels.append(int(row['label']))

        # Odpalamy magię Albumentations (obraz + boxy transformują się razem!)
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)
        img_tensor = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']

        # Konwersja z pascal_voc (piksele z wy-paddowanego obrazu) na format YOLO (0.0 - 1.0)
        yolo_targets = []
        for bbox, cls_id in zip(transformed_bboxes, transformed_labels):
            xmin, ymin, xmax, ymax = bbox
            
            # Skoro zrobiliśmy PadIfNeeded, nasz obraz ma teraz idealnie wymiary (img_size, img_size)
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