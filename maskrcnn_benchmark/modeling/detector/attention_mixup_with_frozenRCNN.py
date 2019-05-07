# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..attentionMixup_module import build_attentionMixup_module


#NOTE(peizhen): generalized rcnn with attention mixup module as its previous component. (only train the attention module)
class AttentionMixupWithFrozenRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(AttentionMixupWithFrozenRCNN, self).__init__()

        #NOTE(peizhen): build attention module...
        self.attention_merger = build_attentionMixup_module(cfg)

        #TODO(peizhen): check. only want attention_merger to learn
        with torch.no_grad():
            self.backbone = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
            # manually set no learn
            self.backbone.requires_grad = False
            self.rpn.requires_grad = False
            self.roi_heads.requires_grad = False

    #NOTE(peizhen): extra feats tensor (8, feat_len, *, *)
    def forward(self, images, feats, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        #NOTE(peizhen): Here outght to be 8 syntheized (stacked on the channel dimension) image tensors but not the original 16 image tensors
        # and the "targes" should be 8 merged annotations tensors. @Pengcheng about this.
        images = to_image_list(images)
        
        #images -> merged_images    
        #NOTE(peizhen): attention mixup module. images (8 x 6 x H x W) -> attention module -> merged_images (8 x 3 x H x W)
        merged_images = self.attention_merger(feats, images)
        features = self.backbone(merged_images.tensors)
        #import ipdb
        #ipdb.set_trace()
        proposals, proposal_losses = self.rpn(merged_images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
