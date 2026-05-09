import torch
import lightning as L
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class PotholeDetector(L.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        # 4 klasy: tło + minor + medium + major
        self.model = fasterrcnn_resnet50_fpn(num_classes=4)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        self.log("train_loss", losses, prog_bar=True)
        return losses

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)