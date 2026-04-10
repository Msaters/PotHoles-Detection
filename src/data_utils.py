import cv2
import os
import torch
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