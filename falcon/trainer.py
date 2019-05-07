# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from apex import amp
from torch.nn import functional as F
import numpy as np

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.image_list import ImageList
from coco_feature_map import get_feature_map


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def _dataset(images, targets, idx, feature_dir):
    # reshape images: [N, 3, H, W] to [N/2, 3*2, H, W]
    _shape = images.tensors.shape
    _images = torch.reshape(images.tensors, [int(_shape[0] / 2), int(_shape[1] * 2), _shape[2], _shape[3]])

    # new image size
    image_size = np.array(images.image_sizes)
    _image_size = []
    for i in range(int(image_size.shape[0] / 2)):
        _image_size.append(image_size[i * 2: i * 2 + 2].max(axis=0).tolist())

    _images = ImageList(_images, _image_size)

    # concat 2 targets to 1 target align to _images
    for i, target in enumerate(targets):
        if 'masks' in target.extra_fields.keys():
            del target.extra_fields['masks']
        if 'keypoints' in target.extra_fields.keys():
            del target.extra_fields['keypoints']
    _targets = []
    for i in range(int(_shape[0] / 2)):
        targets[i * 2].size = targets[i * 2 + 1].size = list(reversed(_image_size[i]))
        _targets.append(cat_boxlist([targets[i * 2], targets[i * 2 + 1]]))

    feats = get_feature_map(idx, feature_dir=feature_dir)
    _shape = feats.shape
    feats = torch.reshape(feats, [int(_shape[0] / 2), int(_shape[1] * 2), _shape[2], _shape[3]])

    return _images, _targets, feats


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        feature_dir
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    #import ipdb
    #ipdb.set_trace()
    for iteration, (images, targets, idx) in enumerate(data_loader, start_iter):
        images, targets, feats = _dataset(images, targets, idx, feature_dir)
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        to_device_time = time.time()
        images = images.to(device)
        ##TODO:
        feats = feats.to(device)
        #import ipdb
        #ipdb.set_trace()
        targets = [target.to(device) for target in targets]
        to_device_time = time.time() - to_device_time

        loss_dict = model(images, feats, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        backward_time = time.time()
        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        backward_time = time.time() - backward_time

        params_update_time = time.time()
        optimizer.step()
        params_update_time = time.time() - params_update_time

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time, device=to_device_time,
                      backward=backward_time, update=params_update_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
