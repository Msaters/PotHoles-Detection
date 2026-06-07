import torch
import lightning as L
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace

class PotholeYoloModule(L.LightningModule):
    def __init__(self, model_cfg: str = 'yolov8n.yaml', lr: float = 1e-3, num_classes: int = 3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        
        # 1. Poprawna inicjalizacja struktury modelu (architektury)
        # Używamy DetectionModel z .yaml, aby fizycznie zbudować warstwy pod 3 klasy
        self.model = DetectionModel(cfg=model_cfg, ch=3, nc=num_classes)
        
        # 2. Transfer Learning: Załadowanie wag z pretrenowanego modelu COCO
        # Chcemy użyć wiedzy modelu (rozpoznawanie krawędzi itp.), ale ignorujemy ostatnią warstwę
        pretrained = YOLO('yolov8n.pt')
        pretrained_state_dict = pretrained.model.state_dict()
        
        # Filtrujemy wagi: kopiujemy tylko te, których kształt się zgadza
        model_state_dict = self.model.state_dict()
        valid_state_dict = {
            k: v for k, v in pretrained_state_dict.items() 
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }
        self.model.load_state_dict(valid_state_dict, strict=False)
        print(f"Załadowano {len(valid_state_dict)}/{len(model_state_dict)} warstw z pretrenowanego modelu.")

        # Odblokowanie wszystkich warstw do treningu
        for param in self.model.parameters():
            param.requires_grad = True

        # 3. Parametry niezbędne dla v8DetectionLoss
        self.model.args = IterableSimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
            tal_topk=10
        )

        self.criterion = v8DetectionLoss(self.model)

    def forward(self, x):
        # W trybie treningowym YOLO zwraca inną strukturę niż w ewaluacji
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
        
        # Obejście dla ewaluacji: YOLO w trybie walidacyjnym zachowuje się inaczej,
        # więc wymuszamy tryb treningowy na czas liczenia funkcji kosztu, 
        # żeby otrzymać "surowe" predykcje potrzebne dla v8DetectionLoss.
        self.model.train() 
        with torch.no_grad():
            preds = self.forward(images)
            loss, loss_items = self.criterion(preds, targets)
        self.model.eval() # Powrót do trybu walidacyjnego
        
        self.log("val/loss", loss.mean(), prog_bar=True, on_epoch=True)
        self.log("val/box_loss", loss_items[0], on_epoch=True)
        self.log("val/cls_loss", loss_items[1], on_epoch=True)
        
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=0.01 # L2 Regularization, zapobiega przeuczeniu
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,   # Łagodniejsze cięcie LR niż poprzednie 0.2
            patience=4, 
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