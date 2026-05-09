import fiddle as fdl
from src.config import schemas
from src.datasets.potholes import PotholeDataModule
from src.models.pothole_detector import PotholeDetector

def build_config():

    model_cfg = fdl.Config(PotholeDetector, lr=1e-4)
    
    datamodule_cfg = fdl.Config(PotholeDataModule, batch_size=4)
    
    training_cfg = fdl.Config(
        schemas.TrainingConfig,
        max_epochs=50,
        wandb_logger=None,
        checkpoint_callback=None,
        callbacks=[]
    )
    
    return fdl.Config(
        schemas.ExperimentConfig,
        name="pothole_detection_v1",
        model=model_cfg,
        data_module=datamodule_cfg,
        training_cfg=training_cfg
    )