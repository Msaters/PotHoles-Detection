import lightning as L
from torch.utils.data import DataLoader, random_split
import kagglehub

from src.data_utils import parse_xmls, PatchPotholeDataset


class PatchPotholeDataModule(L.LightningDataModule):
    """Data module for grid-based pothole detection.
    
    Loads pothole dataset and creates grid-based targets for patch detection.
    Handles train/val/test split and batching.
    
    Args:
        batch_size: Batch size for data loaders (default: 16)
        img_size: Target image size (default: 256)
        grid_size: Output grid dimension (default: 8)
        train_split: Proportion for training set (default: 0.8)
        val_split: Proportion for validation set (default: 0.1)
    """
    
    def __init__(
        self, 
        batch_size: int = 16,
        img_size: int = 256,
        grid_size: int = 8,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.grid_size = grid_size
        self.train_split = train_split
        self.val_split = val_split
        self.data_path = None
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        """Download dataset if needed."""
        self.data_path = kagglehub.dataset_download(
            "idanbaru/annotated-potholes-with-severity-levels"
        )

    def setup(self, stage: str = None):
        """Create train/val/test splits."""
        df = parse_xmls(self.data_path)
        dataset = PatchPotholeDataset(
            df, 
            img_size=self.img_size, 
            grid_size=self.grid_size
        )
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, 
            [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
