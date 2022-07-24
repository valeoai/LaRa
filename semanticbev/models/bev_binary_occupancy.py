from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
import hydra

from semanticbev.utils.losses import SimpleLoss
from semanticbev.utils.wandb_logging import prepare_images_to_log as wandb_prep_images
from semanticbev.utils.tensorboard_logging import prepare_images_to_log as tensorboard_prep_images
from semanticbev.utils.metrics import IntersectionOverUnion


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

        self.metric_iou_val = IntersectionOverUnion(None)

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

        loss = self.loss_fn(preds, batch['binimg']) * preds.shape[0]  # loss_fn is an average -> multiply by batch size

        self.metric_iou_val(preds.sigmoid(), batch['binimg'].int())

        metrics = {'loss': loss.item()}

        images = self.prep_images('val', batch, batch['binimg'], preds.sigmoid(), batch_idx, log_images_interval=200)

        return {'metrics': metrics, 'images': images}

    def validation_epoch_end(self, validation_step_outputs):

        self.log_images(validation_step_outputs)

        metrics = defaultdict(list)
        for output in validation_step_outputs:
            for k, v in output['metrics'].items():
                metrics[k].append(v)  # at this point v is a float
        metrics = {k: torch.tensor(v) for k, v in metrics.items()}

        loss = metrics['loss'].mean()
        # iou = metrics['intersect'].sum() / metrics['union'].sum()

        log_dict = {'val/loss': loss}

        scores = self.metric_iou_val.compute()
        if not isinstance(scores, list):
            scores = [scores]
        for key, value in zip(['vehicles'], scores):
            log_dict[f'val/iou_{key}'] = value
            self.log(f'val_iou_{key}', value, prog_bar=True, logger=False) # ModelCheckpoint monitor
        self.metric_iou_val.reset()

        self.log_dict(log_dict)
        self.log('val_loss', loss, prog_bar=False, logger=False)  # ModelCheckpoint monitor

    def prep_images(self, learning_phase, batch, binimg, preds, batch_idx, log_images_interval):
        if self.logger_name == WANDB_LOGGER_KEY:
            images = wandb_prep_images(learning_phase, batch, binimg, preds, batch_idx, log_images_interval)
        elif self.logger_name == TENSORBOARD_LOGGER_KEY:
            images = tensorboard_prep_images(learning_phase, batch, binimg, preds, batch_idx, log_images_interval)
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
