import fiddle as fdl
from src.config import schemas
from src.datasets.patch_potholes import PatchPotholeDataModule
from src.models.pothole_patch_detector import PotholePatchDetector


def build_config():
    """Build configuration for baseline grid-based pothole detection.
    
    This configuration trains a simple CNN that predicts pothole presence
    in an 8x8 grid of image patches. It serves as a baseline for more
    sophisticated detection approaches.
    """
    
    # Model configuration
    model_cfg = fdl.Config(
        PotholePatchDetector, 
        lr=1e-3,
        grid_size=8
    )
    
    # Data module configuration
    datamodule_cfg = fdl.Config(
        PatchPotholeDataModule,
        batch_size=16,
        img_size=256,
        grid_size=8,
        train_split=0.8,
        val_split=0.1
    )
    
    # Training configuration
    training_cfg = fdl.Config(
        schemas.TrainingConfig,
        max_epochs=30,
        wandb_logger=None,
        checkpoint_callback=None,
        callbacks=[]
    )
    
    # Experiment configuration
    return fdl.Config(
        schemas.ExperimentConfig,
        name="pothole_detection_baseline_grid",
        model=model_cfg,
        data_module=datamodule_cfg,
        training_cfg=training_cfg
    )
