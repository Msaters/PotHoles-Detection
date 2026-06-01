import warnings
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.nms import non_max_suppression

class PotholeYoloModule(L.LightningModule):

    def __init__(self, model_cfg: str = 'yolov8n.yaml', lr: float = 1e-3, num_classes: int = 1):
        super().__init__()
        self.save_hyperparameters()

        self.model = self._build_model(model_cfg, num_classes)
        self.num_classes = num_classes

        self.model.args = IterableSimpleNamespace(
            box=7.5,
            cls=0.5,
            dfl=1.5,
            tal_topk=10,
        )

        if isinstance(self.model, DetectionModel):
            criterion_model = self.model
        elif hasattr(self.model, 'model') and isinstance(self.model.model, DetectionModel):
            criterion_model = self.model.model
        else:
            criterion_model = self.model

        self.criterion = v8DetectionLoss(criterion_model)
        self.lr = lr

        self._reset_validation_tracking()

    def _build_model(self, model_cfg: str, num_classes: int):
        if isinstance(model_cfg, str) and model_cfg.endswith('.pt'):
            try:
                model = YOLO(model_cfg).model
                names = {i: str(i) for i in range(num_classes)}
                model.names = names
                warnings.warn(
                    "YOLO .pt model provided; loaded legacy .pt DetectionModel for backward compatiblity with saved checkpoints. "
                    "If you want class reconfiguration, pass a YAML config path instead."
                )
                return model
            except Exception as exc:
                warnings.warn(
                    f"Unable to instantiate legacy .pt DetectionModel ({model_cfg}); falling back to YAML config. Error: {exc}"
                )
                model_cfg = model_cfg[:-3] + '.yaml'

        model = DetectionModel(model_cfg, nc=num_classes)
        model.names = {i: str(i) for i in range(num_classes)}
        return model

    def _reset_validation_tracking(self):
        self._val_tp = 0
        self._val_fp = 0
        self._val_fn = 0
        self._val_examples = None

    def _xywhn_to_xyxy(self, boxes: torch.Tensor, img_size: int) -> torch.Tensor:
        x_center, y_center, width, height = boxes.unbind(-1)
        x1 = (x_center - width / 2.0) * img_size
        y1 = (y_center - height / 2.0) * img_size
        x2 = (x_center + width / 2.0) * img_size
        y2 = (y_center + height / 2.0) * img_size
        return torch.stack([x1.clamp(0, img_size), y1.clamp(0, img_size), x2.clamp(0, img_size), y2.clamp(0, img_size)], dim=-1)

    def _prepare_targets(self, targets: dict[str, torch.Tensor], img_size: int, batch_size: int) -> list[dict[str, torch.Tensor]]:
        output = []
        for image_idx in range(batch_size):
            mask = targets['batch_idx'] == image_idx
            boxes = targets['bboxes'][mask]
            labels = targets['cls'][mask].long()
            if boxes.numel() == 0:
                output.append({
                    'boxes': torch.zeros((0, 4), device=boxes.device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=boxes.device),
                })
                continue
            output.append({
                'boxes': self._xywhn_to_xyxy(boxes, img_size),
                'labels': labels,
            })
        return output

    def _decode_detections(self, preds):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[0]
        return non_max_suppression(preds, conf_thres=0.1, iou_thres=0.45, max_det=300, nc=self.num_classes)

    def _accumulate_detection_metrics(self, detections: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        for pred, target in zip(detections, targets):
            if pred is None or pred.numel() == 0:
                self._val_fn += int(target['boxes'].shape[0])
                continue

            if target['boxes'].numel() == 0:
                self._val_fp += int(pred.shape[0])
                continue

            pred = pred[pred[:, 4].argsort(descending=True)]
            matched = torch.zeros(target['boxes'].shape[0], dtype=torch.bool, device=pred.device)

            for row in pred:
                ious = box_iou(row[:4].unsqueeze(0), target['boxes']).squeeze(0)
                best_iou, best_index = ious.max(0)
                if best_iou >= 0.5 and not matched[best_index] and row[5].long() == target['labels'][best_index]:
                    self._val_tp += 1
                    matched[best_index] = True
                else:
                    self._val_fp += 1

            self._val_fn += int((~matched).sum())

    def _log_validation_metrics(self):
        precision = self._val_tp / max(self._val_tp + self._val_fp, 1)
        recall = self._val_tp / max(self._val_tp + self._val_fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        self.log('val/detection_precision', precision, prog_bar=True, on_epoch=True)
        self.log('val/detection_recall', recall, prog_bar=True, on_epoch=True)
        self.log('val/detection_f1', f1, prog_bar=True, on_epoch=True)

    def _save_validation_visualization(self, epoch: int):
        if self._val_examples is None:
            return

        images, detections, targets = self._val_examples
        image_count = min(len(images), len(detections), len(targets), 4)
        if image_count == 0:
            return

        fig, axs = plt.subplots(1, image_count, figsize=(5 * image_count, 5))
        if image_count == 1:
            axs = [axs]

        for idx in range(image_count):
            img = images[idx].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            axs[idx].imshow(img)
            axs[idx].axis('off')
            axs[idx].set_title(f'Val sample {idx}')

            for box in targets[idx]['boxes'].cpu().numpy():
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', facecolor='none', linewidth=2)
                axs[idx].add_patch(rect)

            if detections[idx] is not None and detections[idx].numel() > 0:
                for box in detections[idx].cpu().numpy():
                    x1, y1, x2, y2 = box[:4]
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=1.5)
                    axs[idx].add_patch(rect)

        save_dir = Path(self.trainer.log_dir if self.trainer is not None else Path.cwd())
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'val_detection_examples_epoch_{epoch}.png'
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def forward(self, x):
        if isinstance(self.model, DetectionModel):
            base_model = self.model
        elif hasattr(self.model, 'model') and isinstance(self.model.model, DetectionModel):
            base_model = self.model.model
        else:
            base_model = self.model
        return base_model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)

        loss, loss_items = self.criterion(preds, targets)

        self.log('train/loss', loss.mean(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/box_loss', loss_items[0], on_epoch=True)
        self.log('train/cls_loss', loss_items[1], on_epoch=True)
        self.log('train/dfl_loss', loss_items[2], on_epoch=True)

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)

        loss, loss_items = self.criterion(preds, targets)
        self.log('val/loss', loss.mean(), prog_bar=True, on_epoch=True)

        detections = self._decode_detections(preds)
        img_size = images.shape[-1]
        ground_truths = self._prepare_targets(targets, img_size, images.shape[0])
        self._accumulate_detection_metrics(detections, ground_truths)

        if batch_idx == 0:
            self._val_examples = (
                images.detach().cpu(),
                [d.cpu() if d is not None else torch.zeros((0, 6)) for d in detections[:4]],
                ground_truths[:4],
            )

        return loss.mean()

    def on_validation_epoch_start(self):
        self._reset_validation_tracking()

    def on_validation_epoch_end(self):
        self._log_validation_metrics()
        self._save_validation_visualization(self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            # verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
            },
        }

    def on_train_start(self):
        print(f'Trening rozpoczęty na urządzeniu: {self.device}')