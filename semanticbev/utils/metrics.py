from typing import Optional, List

import torch
from torchmetrics.metric import Metric

from semanticbev.models.components.positional_encodings import meshgrid
from einops import rearrange, repeat



def get_str_interval(interval):
    str_interval = [str(int(e*100)) for e in interval]
    str_interval = '-'.join(str_interval)
    return str_interval

class BCEMetric(Metric):
    """
    Computes BCE as a metric
    """
    def __init__(self, pos_weight: float):
        super().__init__()

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='sum')

        self.add_state('loss', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('n_obs', default=torch.zeros(1), dist_reduce_fx='sum')

    def update(self, pred, label):
        self.loss += self.loss_fn(pred, label)
        self.n_obs += pred.numel()

    def compute(self):
        return self.loss / self.n_obs


class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=0.5):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.thresholds = thresholds

        self.add_state('tp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(1), dist_reduce_fx='sum')

    def update(self, pred, label):
        pred = pred.detach().reshape(-1)
        label = label.detach().bool().reshape(-1)

        pred = pred[:, None] >= self.thresholds
        label = label[:, None]

        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)

    def compute(self):
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        return ious


class IntersectionOverInstanceMetric(BaseIoUMetric):
    def __init__(self, thresholds: float):
        super().__init__(thresholds)

    def update(self, pred, label, *args):

        mask = label > 0

        pred = pred[mask]
        label = label[mask]

        return super().update(pred, label)


class IoUMetric(BaseIoUMetric):
    def __init__(self, thresholds: float, min_visibility: Optional[int] = None):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples
        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__(thresholds)

        self.min_visibility = min_visibility

    def update(self, pred, label, visibility):

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility

            pred = pred[mask]
            label = label[mask]

        return super().update(pred, label)


class IoUMetricPerDistance(BaseIoUMetric):
    def __init__(self, thresholds: float, min_visibility: Optional[int] = None, normalized_interval=(0., 1.)):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples
        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__(thresholds)

        self.normalized_inerval = normalized_interval

        self.min_visibility = min_visibility

    def update(self, pred, label, visibility):
        b = pred.shape[0]

        mask = meshgrid(pred.shape[-2:], device=pred.device).pow(2).sum(dim=-1).sqrt()
        mask = (mask >= self.normalized_inerval[0]) & (mask < self.normalized_inerval[1])

        mask = rearrange(mask, '... -> 1 1 ...')
        mask = repeat(mask, '1 ... -> b ...', b=b)

        if self.min_visibility is not None:
            mask = mask & (visibility >= self.min_visibility)

            pred = pred[mask]
            label = label[mask]

        return super().update(pred, label)