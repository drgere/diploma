import math
import torch
from pathlib import Path
from torch import nn as nn
from torch.nn import functional as F
from algo_model.include.defmod import defmod_transformer, def_rnn
from algo_model.include.embedding import embedding
from torch.optim.lr_scheduler import ReduceLROnPlateau


def sched(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):

    def lambda_lr(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr, last_epoch)


def loss_aggregation(loss, aggregation="mean"):
    return (
        loss.mean()
        if aggregation == "mean"
        else loss.sum()
        if aggregation == "sum"
        else loss
    )


def get_model(model, *args, **kwargs):
    mlo = str(model).strip().lower()
    if mlo in MODELS.keys():
        return MODELS[mlo](*args, **kwargs)
    else:
        location = kwargs['device'] if 'device' in kwargs.keys() else 'cuda'
        print(location)
        if Path(model).is_file:
            return torch.load(model, map_location=location)
    raise ValueError("Given model does not exists: {}".format(model))


def linear_comb(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def sched_plateau(optimizer):
    return ReduceLROnPlateau(optimizer, factor=0.1, patience=5, threshold=0.001, cooldown=1)


class cross_entropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, aggregation="mean", ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.aggregation = aggregation
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = loss_aggregation(-log_preds.sum(dim=-1), self.aggregation)
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )
        return linear_comb(loss / n, nll, self.epsilon)


MODELS = {
    'defmod-transformer': defmod_transformer,
    'defmod-rnn': def_rnn,
    'embed2embed-mlp': embedding,
}