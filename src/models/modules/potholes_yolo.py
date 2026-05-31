import torch
import lightning as L
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace

class PotholeYoloModule(L.LightningModule):

    def __init__(self, model_cfg: str = 'yolov8n.yaml', lr: float = 1e-3, num_classes: int = 1):
        super().__init__()
        self.save_hyperparameters()
        
        yolo_model = YOLO(model_cfg)
        self.model = yolo_model.model
        self.model.nc = num_classes

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.args = IterableSimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
            tal_topk=10
        )

        self.criterion = v8DetectionLoss(self.model)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        
        loss, loss_items = self.criterion(preds, targets)
        
        self.log("train/loss", loss.mean(), prog_bar=True, on_step=True, on_epoch=True)
        
        self.log("train/box_loss", loss_items[0], on_epoch=True)
        self.log("train/cls_loss", loss_items[1], on_epoch=True)
        self.log("train/dfl_loss", loss_items[2], on_epoch=True)
        
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        
        loss, loss_items = self.criterion(preds, targets)
        
        self.log("val/loss", loss.mean(), prog_bar=True, on_epoch=True)
        return loss.mean()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.2, 
            patience=5, 
            #verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }

    def on_train_start(self):

        print(f"Trening rozpoczęty na urządzeniu: {self.device}")