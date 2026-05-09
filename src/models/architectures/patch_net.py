import torch
import torch.nn as nn


class PotholePatchNet(nn.Module):
    """Baseline CNN for grid-based pothole detection.
    
    This architecture detects potholes by dividing the image into an 8x8 grid
    and predicting presence/absence of potholes in each grid cell.
    
    The network progressively downsamples the input image from 256x256 to 8x8,
    matching the grid output size, then uses 1x1 convolutions for final 
    classification per grid cell.
    
    Input:  (B, 3, 256, 256) - RGB images
    Output: (B, 1, 8, 8) - Binary predictions per grid cell
    """
    
    def __init__(self):
        super(PotholePatchNet, self).__init__()
        
        # Feature extraction: progressively downsample from 256x256 to 8x8
        self.features = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 128x128
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 16x16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 8x8
        )
        
        # Classification head: 1x1 convolutions for per-cell predictions
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)  # Binary output per cell
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 256, 256)
        
        Returns:
            Output tensor of shape (B, 1, 8, 8) with logits for each grid cell
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
