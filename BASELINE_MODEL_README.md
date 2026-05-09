# Baseline Grid-Based Pothole Detector

## Overview

This is a baseline convolutional neural network for detecting potholes using a grid-based approach. The model divides images into an 8×8 grid and predicts the presence or absence of potholes in each grid cell.

## Architecture

### Model: `PotholePatchNet`

```
Input: (B, 3, 256, 256) - RGB images

Feature Extraction (5 conv blocks):
- Conv2d(3, 16) → ReLU → MaxPool(2,2)    # 256×256 → 128×128
- Conv2d(16, 32) → ReLU → MaxPool(2,2)   # 128×128 → 64×64
- Conv2d(32, 64) → ReLU → MaxPool(2,2)   # 64×64 → 32×32
- Conv2d(64, 128) → ReLU → MaxPool(2,2)  # 32×32 → 16×16
- Conv2d(128, 256) → ReLU → MaxPool(2,2) # 16×16 → 8×8

Classification Head:
- Conv2d(256, 128, kernel=1) → ReLU
- Conv2d(128, 1, kernel=1)

Output: (B, 1, 8, 8) - Binary predictions per grid cell
```

**Total Parameters**: ~600,000

## Dataset

### `PatchPotholeDataset`

Transforms bounding box annotations into grid-based targets:

1. Loads images from file paths
2. Resizes to 256×256
3. Normalizes pixel values to [0, 1]
4. Creates 8×8 grid targets
5. Marks cells containing pothole bounding boxes with 1.0, others with 0.0

**Input**: DataFrame with columns:
- `file`: Path to image
- `xmin`, `ymin`, `xmax`, `ymax`: Bounding box coordinates

## Project Structure

```
src/
├── models/
│   ├── pothole_patch_detector.py      # Lightning module for training
│   ├── architectures/
│   │   └── patch_net.py               # PotholePatchNet architecture
│   └── pothole_detector.py            # Original Faster RCNN detector
├── datasets/
│   └── patch_potholes.py              # PatchPotholeDataModule
├── data_utils.py                      # PatchPotholeDataset + parse_xmls
└── config/
    └── potholes_baseline.py           # Fiddle configuration

notebooks/
└── 03_baseline_grid_detector.ipynb    # Full training notebook

scripts/
└── train_baseline.py                  # Standalone training script
```

## Usage

### Option 1: Use the Notebook

Open `notebooks/03_baseline_grid_detector.ipynb` and run all cells:

```python
# Key sections:
# 1. Import libraries
# 2. Define PatchPotholeDataset
# 3. Define PotholePatchNet
# 4. Load data and initialize model
# 5. Training loop (10 epochs)
# 6. Visualization of training history
# 7. Evaluation on test set
# 8. Visualization of predictions
# 9. Save/load model checkpoint
```

### Option 2: Use the Standalone Script

```bash
python scripts/train_baseline.py
```

This runs a complete training pipeline with:
- Dataset loading and splitting (80/10/10)
- Model initialization
- 10 epochs of training
- Evaluation on test set

### Option 3: Use PyTorch Lightning with Configuration

```python
from src.config.potholes_baseline import build_config
import fiddle as fdl

cfg = build_config()
built_cfg = fdl.build(cfg)

# Use the configuration-based training pipeline
python scripts/train_model.py src/config/potholes_baseline.py
```

## Training Details

### Configuration
- **Batch Size**: 16
- **Image Size**: 256×256
- **Grid Size**: 8×8
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Loss Function**: BCEWithLogitsLoss
- **Epochs**: 10-30 (configurable)
- **Data Split**: 80% train, 10% val, 10% test

### Metrics
- **Loss**: Binary Cross-Entropy with Logits
- **Accuracy**: Proportion of correctly classified grid cells
- **Threshold**: 0.5 for converting logits to binary predictions

## Integration with Project

The baseline model is fully integrated with the existing project structure:

1. **Lightning Module**: `PotholePatchDetector` follows the same pattern as `PotholeDetector`
2. **Configuration**: Uses Fiddle for hyperparameter management
3. **Data Loading**: Leverages existing `parse_xmls()` utility
4. **Modular Design**: Architecture separable in `patch_net.py`

## Key Advantages

✓ **Simple baseline**: Easy to understand and modify
✓ **Grid-based**: Localizes pothole regions without bounding box regression
✓ **Efficient**: Single-stage detection with no NMS required
✓ **Scalable**: Integrates with PyTorch Lightning for distributed training
✓ **Well-documented**: Comprehensive notebook and inline comments

## Limitations & Future Work

### Limitations
- Grid size fixed to 8×8 (may miss small/large potholes)
- No severity level prediction
- Binary classification only (presence/absence)
- No temporal information (video)

### Improvements
- [ ] Multi-scale grid predictions
- [ ] Severity classification per grid cell
- [ ] Data augmentation (rotation, brightness, etc.)
- [ ] Deeper architectures (ResNet backbone)
- [ ] Attention mechanisms
- [ ] Post-processing with CRF
- [ ] Uncertainty estimation

## Files Created/Modified

### New Files
1. `src/models/architectures/patch_net.py` - PotholePatchNet architecture
2. `src/models/pothole_patch_detector.py` - Lightning module
3. `src/datasets/patch_potholes.py` - Data module with train/val/test split
4. `src/config/potholes_baseline.py` - Fiddle configuration
5. `notebooks/03_baseline_grid_detector.ipynb` - Training notebook
6. `scripts/train_baseline.py` - Standalone training script

### Modified Files
1. `src/data_utils.py` - Added `PatchPotholeDataset` class

## Performance Benchmarks

Expected performance on test set (after 10 epochs):
- **Accuracy**: ~85-90% (cell-level)
- **Loss**: ~0.15-0.25 (BCEWithLogitsLoss)

*Note: Actual performance depends on data quality and hardware*

## References

- Original pothole dataset: https://www.kaggle.com/datasets/idanbaru/annotated-potholes-with-severity-levels
- PyTorch Grid-based detection: Common in YOLO, YOLOv2 architectures
- Grid heatmap approach: Similar to crowd counting and density estimation tasks
