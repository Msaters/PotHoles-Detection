import torch
import torch.nn as nn
import lightning as L

from src.models.architectures.patch_net import PotholePatchNet


class PotholePatchDetector(L.LightningModule):
    """Lightning module for baseline grid-based pothole detection.
    
    Uses a CNN that predicts pothole presence in an 8x8 grid of image patches.
    Each grid cell is classified as containing a pothole (1) or not (0).
    
    Args:
        lr: Learning rate for the optimizer (default: 1e-3)
        grid_size: Size of the output grid (default: 8)
    """
    
    def __init__(self, lr: float = 1e-3, grid_size: int = 8):
        super().__init__()
        self.model = PotholePatchNet()
        self.lr = lr
        self.grid_size = grid_size
        
        # Binary cross-entropy loss for grid cell classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, 3, 256, 256)
        
        Returns:
            Grid predictions of shape (B, 1, 8, 8)
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = self.criterion(outputs, targets)
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = self.criterion(outputs, targets)
        
        # Calculate accuracy: proportion of correctly predicted grid cells
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions == targets).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        
        loss = self.criterion(outputs, targets)
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions == targets).float().mean()
        
        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        
        return {"test_loss": loss, "test_accuracy": accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
