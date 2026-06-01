import fiddle as fdl
from src.config import schemas
from src.datasets.potholes_yolo import PotholeDataModuleYOLO
from src.models.modules.potholes_yolo import PotholeYoloModule

def build_config():

    model_cfg = fdl.Config(
        PotholeYoloModule,
        lr=1e-3,
        model_cfg='yolov8n.yaml',
        num_classes=3
    )
    
    datamodule_cfg = fdl.Config(
        PotholeDataModuleYOLO, 
        batch_size=16, 
        img_size=640
    )
    
    training_cfg = fdl.Config(
        schemas.TrainingConfig,
        max_epochs=5,
        wandb_logger=None,
        checkpoint_callback=None,
        callbacks=[]
    )
    
    return fdl.Config(
        schemas.ExperimentConfig,
        name="pothole_yolo_detection_v1",
        model=model_cfg,
        data_module=datamodule_cfg,
        training_cfg=training_cfg
    )