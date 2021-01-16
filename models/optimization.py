import math
import warnings
from typing import List
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer_class(opt_type='adam'):
    if opt_type.lower() == 'sgd':
        return torch.optim.SGD
    if opt_type.lower() == 'adam':
        return torch.optim.Adam
    if opt_type.lower() == 'adamw':
        return torch.optim.AdamW
    if opt_type.lower() == 'lookahead':
        return NotImplementedError('Lookahead is not implemented yet')
    return ValueError(f'{opt_type} optimizer is not supported')


def get_scheduler(optimizer, kwargs):
    scheduler_params = kwargs.pop('params')
    scheduler_type = kwargs.pop('type')
    if scheduler_type == 'cosine_anneal':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_type == 'cosine_anneal_simple':
        annealing = lambda x: (((1 + math.cos(x * math.pi / scheduler_params['num_epochs'])) / 2) ** 1.0) * 0.9 + 0.1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=annealing)
    return {
        'scheduler': scheduler,
        **kwargs
    }
    

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    '''
    github.com/PyTorchLightning/PyTorch-Lightning-Bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py
    '''
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]