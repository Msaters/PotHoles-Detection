import lightning as L
from torch.utils.data import DataLoader
import kagglehub
#from .utils import parse_xmls, PotholeDataset
from src.data_utils import parse_xmls, PotholeDataset

class PotholeDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = None

    def prepare_data(self):
        self.data_path = kagglehub.dataset_download("idanbaru/annotated-potholes-with-severity-levels")

    def setup(self, stage=None):
        df = parse_xmls(self.data_path)
        self.train_ds = PotholeDataset(df)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))