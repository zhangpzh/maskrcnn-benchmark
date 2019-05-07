# huangjj-pkusz-pcl
# load the pretrain fpn(it should be a checkpoint, and the last checkpoint is loaded as default) to initialize the mix-up model
import logging
import os

import torch
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format

def get_model_file(path):
    save_file = path
    try:
        with open(save_file, "r") as f:
            last_saved = f.read()
            last_saved = last_saved.strip()
    except IOError:
        # if file doesn't exist, maybe because it has just been
        # deleted by a separate process
        last_saved = ""
    return last_saved


def load_file(cfg, f):
    if f.endswith(".pkl"):
        return load_c2_format(cfg, f)
    return torch.load(f, map_location=torch.device("cpu"))


def load_model(model, checkpoint):
    load_state_dict(model, checkpoint.pop("model"))
    return model


def load_pretrain_detector(cfg, model):
    logger = logging.getLogger(__name__)

    path_to_model = cfg.PRETRAIN_DET_DIR
    save_file = os.path.join(path_to_model, "last_checkpoint")

    #import ipdb
    #ipdb.set_trace()
    if os.path.exists(save_file):
        f = get_model_file(save_file)
    if not f:
        logger.info("No pretrained detecotr found.")
        exit(0)

    checkpoint = load_file(cfg, f)

    return load_model(model, checkpoint)
