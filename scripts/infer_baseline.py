#!/usr/bin/env python3
"""
Inference script for baseline grid-based pothole detector.
Visualizes predictions on images with color-coded grid overlay.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.models.architectures.patch_net import PotholePatchNet


class PotholeDetectorInference:
    """Inference wrapper for pothole detection model."""
    
    def __init__(self, checkpoint_path, device=None):
        """Load trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PotholePatchNet().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Handle Lightning state dicts where weights are prefixed with "model."
            if any(key.startswith('model.') for key in state_dict.keys()):
                stripped_state = {key.replace('model.', '', 1): value for key, value in state_dict.items()}
                self.model.load_state_dict(stripped_state)
            else:
                raise e
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"  Device: {self.device}")
    
    def predict(self, image_path, img_size=256, grid_size=8, threshold=0.5):
        """Run inference on an image.
        
        Args:
            image_path: Path to image file
            img_size: Model input size (default: 256)
            grid_size: Grid size (default: 8)
            threshold: Confidence threshold (default: 0.5)
        
        Returns:
            tuple: (original_image, resized_image, grid_predictions, confidence_map)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Resize for model
        img_resized = cv2.resize(img, (img_size, img_size))
        
        # Normalize and convert to tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(img_tensor)
            confidence_map = torch.sigmoid(output).squeeze().cpu().numpy()
            grid_predictions = (confidence_map > threshold).astype(np.float32)
        
        return img, img_resized, grid_predictions, confidence_map
    
    def visualize(self, image_path, save_path=None, threshold=0.5, alpha=0.4):
        """Visualize predictions on image with color overlay.
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization (optional)
            threshold: Prediction threshold (default: 0.5)
            alpha: Overlay transparency (default: 0.4)
        """
        # Get predictions
        img, img_resized, grid_pred, conf_map = self.predict(
            image_path, 
            threshold=threshold
        )
        
        grid_size = grid_pred.shape[-1]
        img_size = img_resized.shape[0]
        cell_size = img_size / grid_size
        
        # Create colored overlay
        overlay = img_resized.copy()
        
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = int(j * cell_size)
                y1 = int(i * cell_size)
                x2 = int((j + 1) * cell_size)
                y2 = int((i + 1) * cell_size)
                
                if grid_pred[i, j] == 1.0:
                    # Pothole (red)
                    color = (255, 100, 100)  # Light red in BGR
                else:
                    # Background (blue)
                    color = (100, 100, 255)  # Light blue in BGR
                
                # Draw filled rectangle with transparency
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                # Draw border
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Blend original and overlay
        result = cv2.addWeighted(img_resized, 1 - alpha, overlay, alpha, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Original image
        axes[0].imshow(img_resized)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Confidence map (heatmap)
        im = axes[1].imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1].set_title("Confidence Map", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Confidence')
        
        # Result with overlay
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        axes[2].imshow(result_rgb)
        axes[2].set_title("Predictions (Red=Background, Blue=Pothole)", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(1, 0.39, 0.39), label='No pothole'),
            Patch(facecolor=(0.39, 0.39, 1), label='Pothole detected')
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        num_potholes = np.sum(grid_pred)
        total_cells = grid_pred.size
        percentage = (num_potholes / total_cells) * 100
        
        print(f"\n{'Inference Results':^50}")
        print("-" * 50)
        print(f"Image: {Path(image_path).name}")
        print(f"Grid size: {grid_size}×{grid_size}")
        print(f"Pothole cells detected: {int(num_potholes)}/{total_cells}")
        print(f"Coverage: {percentage:.1f}%")
        print(f"Average confidence: {np.mean(conf_map):.4f}")
        print("-" * 50)
        
        return result_rgb, grid_pred, conf_map


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inference script for pothole detector baseline model"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str,
        default="checkpoints/pothole_patch_net_baseline.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save visualization"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold (default: 0.5)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Overlay transparency (default: 0.4)"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Make sure you've trained the model first: python scripts/train_baseline.py")
        return
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image not found: {args.image}")
        return
    
    # Run inference
    detector = PotholeDetectorInference(args.checkpoint)
    detector.visualize(args.image, save_path=args.output, threshold=args.threshold, alpha=args.alpha)


if __name__ == "__main__":
    main()
