from configs.config import ConfigReader
from models.effdet import DetectionModel
from data.detection_dataset import get_coco_loaders
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datetime import datetime
import pytorch_lightning as pl
import argparse
import os


parser = argparse.ArgumentParser(description="Train an object detection model")
parser.add_argument("-cfg", type=str, default="/configs/effdet_train.yaml")
parser.add_argument("-root", type=str, default=".")
args = parser.parse_args()


def train():
    configuration = ConfigReader(args.root).process(args.cfg)

    checkpoint_callback = ModelCheckpoint(
        os.path.join(
            './checkpoints/',
            configuration['experiment_name'] + '_' + datetime.now().strftime('%m.%d_%H.%M'), 
            "{epoch}"
        ), monitor="val_loss"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(
        'logs',
        name=configuration['task']+'_'+configuration['experiment_name']
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[lr_monitor],
        checkpoint_callback=checkpoint_callback,
        max_epochs=configuration['process']['max_epochs'],
        precision=configuration['process']['precision'],
        gpus=configuration['process']['gpus'],
        check_val_every_n_epoch=5,
    )

    if configuration['process']['checkpoint'] is not None:
        model = DetectionModel.load_from_checkpoint(
            configuration['process']['checkpoint'],
            main_config = configuration['model']
        )
    else:
        model = DetectionModel(configuration['model'])
        
    train_loader, val_loader = get_coco_loaders(
        configuration['data']['train'],
        configuration['data']['val'],
        batch_size=(
            configuration['process']['batch_size'], configuration['process']['batch_size']
        ),
        crop_size = configuration['data']['crop_size']
    )

    trainer.fit(
        model, train_dataloader = train_loader, val_dataloaders = val_loader
    )

    return trainer


if __name__ == "__main__":
    train()
