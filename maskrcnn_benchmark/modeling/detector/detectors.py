# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
#NOTE(peizhen)
from .attention_mixup_with_frozeRCNN import AttentionMixupWithFrozenRCNN 


#_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
		"AttentionMixupWithFrozenRCNN": AttentionMixupWithFrozenRCNN} #NOTE(peizhen)


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
