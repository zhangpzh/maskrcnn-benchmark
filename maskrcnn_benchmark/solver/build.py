# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR


#def make_optimizer(cfg, model):
#    params = []
#    for key, value in model.named_parameters():
#        if not value.requires_grad:
#            print('%s: %s does not require gradient !'%(key, str(value.shape)))
#            continue
#        print('%s: %s needs gradient !'%(key, str(value.shape)))
#        lr = cfg.SOLVER.BASE_LR
#        weight_decay = cfg.SOLVER.WEIGHT_DECAY
#        if "bias" in key:
#            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
#            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
#        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
#
#    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
#    return optimizer

#TODO(peizhen): only attention_merger module learns
def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not key.startswith('attention_merger'):
            value.requires_grad = False

    for key, value in model.named_parameters():
        if not value.requires_grad:
            print('%s: %s does not require gradient !'%(key, str(value.shape)))
            continue
        print('%s: %s needs gradient !'%(key, str(value.shape)))
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
