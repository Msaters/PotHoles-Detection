#!/usr/bin/env python3
"""
Training script for baseline grid-based pothole detector.

This is a simplified example showing how to train the patch-based pothole detector.
For production use, use the more sophisticated training pipeline in scripts/train_model.py
with the configuration in src/config/potholes_baseline.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import lightning as L
from torch.utils.data import DataLoader
import kagglehub

from src.data_utils import parse_xmls, PatchPotholeDataset
from src.models.pothole_patch_detector import PotholePatchDetector


def train_baseline():
    """Train the baseline pothole patch detector."""
    
    # Set random seed for reproducibility
    L.seed_everything(42)
    
    # Configuration
    BATCH_SIZE = 16
    IMG_SIZE = 256
    GRID_SIZE = 8
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 30
    
    # Device - automatically detect GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU detected, using CPU (slower training)")
    
    # 1. Load dataset
    print("Loading dataset...")
    data_path = kagglehub.dataset_download("idanbaru/annotated-potholes-with-severity-levels")
    df = parse_xmls(data_path)
    print(f"Loaded {len(df)} annotations from {len(df['file'].unique())} images")
    
    # 2. Create dataset and dataloaders
    print("Creating dataset and dataloaders...")
    dataset = PatchPotholeDataset(df, img_size=IMG_SIZE, grid_size=GRID_SIZE)
    
    # Split dataset: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    from torch.utils.data import random_split
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    # Use pin_memory if GPU is available
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    
    # Create test dataloader
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    
    # 3. Initialize model
    print("Initializing model...")
    model = PotholePatchDetector(lr=LEARNING_RATE, grid_size=GRID_SIZE)
    
    # 4. Setup trainer
    print("Setting up trainer...")
    # Use GPU if available, fallback to CPU
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1 if torch.cuda.is_available() else "auto"
    
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # 5. Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # 6. Test
    print("Testing model...")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    trainer.test(model, test_loader)
    
    # 7. Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "pothole_patch_net_baseline.pt"
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'hyperparameters': {
            'img_size': IMG_SIZE,
            'grid_size': GRID_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'max_epochs': MAX_EPOCHS,
        }
    }, checkpoint_path)
    print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    print("Training completed!")
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = train_baseline()
