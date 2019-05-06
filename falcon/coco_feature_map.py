import argparse
import os

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


class ResNet50FeatureMap(torch.nn.Module):
    def __init__(self):
        super(ResNet50FeatureMap, self).__init__()
        layer4 = list(torchvision.models.resnet50(pretrained=True).children())[:-2]
        self.model = torch.nn.Sequential(*layer4)

    def forward(self, image):
        return self.model(image)


def get_feature_map(image_ids, feature_dir=""):
    if not isinstance(image_ids, (list, tuple)):
        image_ids = [image_ids]
    res = np.load(os.path.join(feature_dir, str(image_ids[0]) + ".npy"))[np.newaxis, :]
    if len(image_ids) == 1:
        return torch.from_numpy(res)

    #for _, _id in enumerate(image_ids, 1):
    for _id in image_ids[1:]:
        res = np.concatenate(
            (res, np.load(os.path.join(feature_dir, str(_id) + ".npy"))[np.newaxis, :]))
    return torch.from_numpy(res)


def main():
    # Demo command as follow:
    # export NGPUS=8
    # python -m torch.distributed.launch --nproc_per_node=$NGPUS falcon/coco_feature_map.py \
    # --config-file "configs/e2e_faster_rcnn_R_50_FPN_1x.yaml" \
    # --feature-map-size 25 35 DATASETS.TEST "('coco_2014_train', 'coco_2014_valminusminival')" \
    # TEST.IMS_PER_BATCH 64 OUTPUT_DIR coco_featuremap_resnet50

    parser = argparse.ArgumentParser(description="Generate coco dataset feature map")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--feature-map-size", type=int, nargs=2, default=(25, 35))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("falcon.coco_feature_map", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = ResNet50FeatureMap()
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")
    model = model.to(device)
    logger.info(model)

    output_dir = cfg.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    data_loaders = make_data_loader(
        cfg,
        is_train=False,
        is_distributed=args.distributed
    )

    model.eval()
    for data_loader in data_loaders:
        for iteration, (images, targets, idx) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            with torch.no_grad():
                output = model(images.tensors)
                output = F.interpolate(output, size=args.feature_map_size,
                                       mode='bilinear', align_corners=True)
                torch.cuda.synchronize()
                feats = [item.to(cpu_device) for item in output]
            logger.info("=== idx: {}".format(idx))
            for _id, feat in zip(idx, feats):
                np.save(os.path.join(output_dir, str(_id) + ".npy"), feat.detach().numpy())


if __name__ == '__main__':
    main()
