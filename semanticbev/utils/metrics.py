from typing import Optional

import torch
from torchmetrics.metric import Metric
from torchmetrics.functional import stat_scores


class IntersectionOverUnion(Metric):
    """Computes intersection-over-union."""
    def __init__(
        self,
        n_classes: Optional[int] = None,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.absent_score = absent_score

        self.buffer_size = 1 if n_classes is None else n_classes

        self.add_state('true_positive', default=torch.zeros(self.buffer_size), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(self.buffer_size), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(self.buffer_size), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(self.buffer_size), dist_reduce_fx='sum')

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        tps, fps, _, fns, sups = stat_scores(prediction, target, threshold=self.threshold, num_classes=self.n_classes)

        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):

        scores = torch.zeros(self.buffer_size, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.buffer_size):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # If this class is absent in the target (no support) AND absent in the pred (no true or false
            # positives), then use the absent_score for this class.
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # Remove the ignored class index from the scores.
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.buffer_size):
            scores = torch.cat([scores[:self.ignore_index], scores[self.ignore_index+1:]])

        return scores