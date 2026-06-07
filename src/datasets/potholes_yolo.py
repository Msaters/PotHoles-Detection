import torch
import lightning as L
from torch.utils.data import DataLoader
import kagglehub
from src.data_utils import parse_xmls, PotholeDatasetYolo

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
        self.train_ds = PotholeDatasetYolo(df, img_size=self.img_size)

    def collate_fn(self, batch):
        """
        Łączy pojedyncze przykłady w batch, którego spodziewa się v8DetectionLoss.
        """
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
            num_workers=8
        )

    def val_dataloader(self):

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=4
        )