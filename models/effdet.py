import pytorch_lightning as pl
from effdet.bench import _batch_detection, _post_process, Anchors, AnchorLabeler, DetectionLoss
from torch import nn
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from .optimization import get_optimizer_class, get_scheduler
import torch


class DetectionModel(pl.core.lightning.LightningModule):
    def __init__(self, main_config):
        super(DetectionModel, self).__init__()
        # optimizer
        self.lr = main_config['optimizer']['lr']
        self.opt_class = get_optimizer_class(main_config['optimizer']['type'])
        self.scheduler_config = main_config['optimizer']['scheduler']
        # efficientdet config
        self.config = get_efficientdet_config(main_config['type'])
        self.config.image_size = main_config['image_size']
        self.config.num_classes = main_config['num_classes']
        self.config.max_det_per_image = main_config['max_det_per_image']
        # anchors
        self.anchors = Anchors.from_config(self.config)
        self.anchor_labeler = AnchorLabeler(
            self.anchors, self.config.num_classes, match_threshold=0.5
        )
        # effdet model itself
        self.model = EfficientDet(self.config, pretrained_backbone=main_config['pretrained'])
        self.model.reset_head(num_classes=self.config.num_classes)
        self.model.class_net = HeadNet(self.config, num_outputs=self.config.num_classes)
        # combined detection loss
        self.loss_fn = DetectionLoss(self.config)

    def configure_optimizers(self):
        optimizer = self.opt_class(
            self.parameters(),
            lr=self.lr
        )
        if self.scheduler_config is not None:
            lr_scheduler = get_scheduler(optimizer, self.scheduler_config)
            return [optimizer], [lr_scheduler]
        return optimizer

    def forward(self, batch):
        class_out, box_out = self.model(batch['image'])
        output = {}
        if 'labels' in batch:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                batch['annotations'], batch['labels']
            )
            loss, class_loss, box_loss = self.loss_fn(
                class_out, box_out, cls_targets, box_targets, num_positives
            )
            output = {'loss': loss, 'class_loss': class_loss, 'box_loss': box_loss}
        if not self.training:
            img_info = {'shape': batch['image'].shape[0], 'img_scale': None, 'img_size': None}
            if 'img_info' in batch:
                img_info['img_scale'] = batch['img_info']['img_scale']
                img_info['img_size'] = batch['img_info']['img_size']
            output['detections'] = self.detect(class_out, box_out, img_info)
        return output

    def detect(self, class_out, box_out, img_info):
        class_out, box_out, indices, classes = _post_process(
            class_out, box_out,
            num_levels=self.config.num_levels,
            num_classes=self.config.num_classes,
            max_detection_points=self.config.max_detection_points
        )
        return _batch_detection(
            img_info['shape'], class_out, box_out,
            self.anchors.boxes, indices, classes,
            img_info['img_scale'], img_info['img_size'],
            max_det_per_image=self.config.max_det_per_image,
            soft_nms=self.config.soft_nms
        )

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        for key in ('loss', 'class_loss', 'box_loss'):
            self.log('train_' + key, outputs[key].item())
        return outputs['loss']

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        for key in ('loss', 'class_loss', 'box_loss'):
            self.log('val_' + key, outputs[key].item())
            
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)
