import matplotlib.pyplot as plt

import torch.nn as nn
import pytorch_lightning as pl
import hydra

from semanticbev.utils.losses import SimpleLoss
from semanticbev.utils.wandb_logging import prepare_images_to_log as wandb_prep_images
from semanticbev.utils.tensorboard_logging import prepare_images_to_log as tensorboard_prep_images
from semanticbev.utils.metrics import BCEMetric, IoUMetric, IoUMetricPerDistance, get_str_interval


TENSORBOARD_LOGGER_KEY = 'tensorboard'
WANDB_LOGGER_KEY = 'wandb'


def compute_iou(preds, binimg):
    """Assumes preds has NOT been sigmoided yet
    """
    pred = (preds.sigmoid() > 0.5)
    tgt = binimg.bool()
    intersect = (pred & tgt).sum().float()
    union = (pred | tgt).sum().float()
    return intersect, union


class BEVBinaryOccupancy(pl.LightningModule):

    def __init__(self, net, pos_weight, optimizer_conf=None, scheduler_conf=None, logger_name='tensorboard'):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf
        self.network = hydra.utils.instantiate(net)

        self.loss_fn = SimpleLoss(pos_weight)
        self.bce_metric = BCEMetric(pos_weight)

        self.metric_iou_vis1 = IoUMetric(thresholds=0.5, min_visibility=1)
        self.metric_iou_vis2 = IoUMetric(thresholds=0.5, min_visibility=2)


        # intervals in percentages (the BEV map is a square so we need >100% to compute metric at corners)
        intervals = [
            [0., 0.1],
            [0.1, 0.2],
            [0.2, 0.4],
            [0.4, 0.6],
            [0.6, 0.8],
            [0.8, 1.],
            [1., 1.5],
        ]
        self.metric_iou_per_distance_vis1 = nn.ModuleDict({
            get_str_interval(interval): IoUMetricPerDistance(thresholds=0.5, min_visibility=1,
                                                             normalized_interval=interval)
            for interval in intervals
        })
        self.metric_iou_per_distance_vis2 = nn.ModuleDict({
            get_str_interval(interval): IoUMetricPerDistance(thresholds=0.5, min_visibility=2,
                                                             normalized_interval=interval)
            for interval in intervals
        })

        self.logger_name = logger_name

    def forward(self, batch):
        preds = self.network(**batch)
        return preds


    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/iou_vehicles": 0})


    def training_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.loss_fn(preds, batch['binimg'])

        intersect, union = compute_iou(preds, batch['binimg'])
        iou = intersect / union if (union > 0) else 1.0
        self.log('iou', iou, prog_bar=True, logger=False)

        self.log_dict({
            'train/loss': loss.item(),
            'train/iou': iou,
            'train/epoch': self.current_epoch
        })

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        preds = self(batch)

        self.bce_metric(preds, batch['binimg'])

        self.metric_iou_vis1(preds.sigmoid(), batch['binimg'], batch['visibility'].int())
        self.metric_iou_vis2(preds.sigmoid(), batch['binimg'], batch['visibility'].int())

        for metric_iou in self.metric_iou_per_distance_vis1.values():
            metric_iou(preds.sigmoid(), batch['binimg'], batch['visibility'].int())
        for metric_iou in self.metric_iou_per_distance_vis2.values():
            metric_iou(preds.sigmoid(), batch['binimg'], batch['visibility'].int())

        images = self.prep_images('val', batch, preds.sigmoid(), batch_idx, log_images_interval=200)

        return {'images': images}

    def validation_epoch_end(self, validation_step_outputs):

        self.log_images(validation_step_outputs)

        loss = self.bce_metric.compute().item()
        self.log('val_loss', loss, prog_bar=False, logger=False)  # ModelCheckpoint monitor

        log_dict = {'val/loss': loss}

        #### SCORES PER VISIBILITY LEVEL
        scores = self.metric_iou_vis1.compute()
        if not isinstance(scores, list):
            scores = [scores]
        for key, value in zip(['vehicles'], scores):
            log_dict[f'val/iou_vis1_{key}'] = value
            self.log(f'val_iou_{key}', value.item(), prog_bar=True, logger=False)  # ModelCheckpoint monitor
        self.metric_iou_vis1.reset()

        scores = self.metric_iou_vis2.compute()
        if not isinstance(scores, list):
            scores = [scores]
        for key, value in zip(['vehicles'], scores):
            log_dict[f'val/iou_vis2_{key}'] = value.item()
        self.metric_iou_vis2.reset()

        #### SCORES PER DISTANCE
        for interval, metric_iou in self.metric_iou_per_distance_vis1.items():
            scores = metric_iou.compute()
            if not isinstance(scores, list):
                scores = [scores]
            for key, value in zip(['vehicles'], scores):
                log_dict[f'val_distances/iou_vis1_{interval}_{key}'] = value.item()
            metric_iou.reset()

        for interval, metric_iou in self.metric_iou_per_distance_vis2.items():
            scores = metric_iou.compute()
            if not isinstance(scores, list):
                scores = [scores]
            for key, value in zip(['vehicles'], scores):
                log_dict[f'val_distances/iou_vis2_{interval}_{key}'] = value.item()
            metric_iou.reset()

        self.log_dict(log_dict)

    def prep_images(self, learning_phase, batch, preds, batch_idx, log_images_interval):
        if self.logger_name == WANDB_LOGGER_KEY:
            images = wandb_prep_images(learning_phase, batch, preds, batch_idx, log_images_interval)
        elif self.logger_name == TENSORBOARD_LOGGER_KEY:
            images = tensorboard_prep_images(learning_phase, batch, preds, batch_idx, log_images_interval)
        else:
            images = {}

        return images

    def log_images(self, outputs):
        aggregated_images = {}
        list_of_images_dict = [output['images'] for output in outputs]
        for images_dict in list_of_images_dict:
            aggregated_images.update(images_dict)

        if self.logger_name == WANDB_LOGGER_KEY:
            self.logger.experiment.log({**aggregated_images})
        elif self.logger_name == TENSORBOARD_LOGGER_KEY:
            for images_title, figure in aggregated_images.items():
                self.logger.experiment.add_figure(images_title, figure, global_step=self.current_epoch)
            plt.close('all')
        elif self.logger_name == '' or self.logger_name is None:
            pass
        else:
            raise NotImplementedError


    def configure_optimizers(self):
        
        if not self.optimizer_conf:
            return None
        
        optimizer = hydra.utils.instantiate(self.optimizer_conf, params=self.parameters())
        
        if not self.scheduler_conf:
            return optimizer
        
        scheduler = hydra.utils.instantiate(self.scheduler_conf, optimizer=optimizer)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,

            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": 'epoch',

            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,

            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",

            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,

            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": 'lr',
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_class_path'] = self.__module__ + '.' + self.__class__.__qualname__
