export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS falcon/train_net.py --config-file "configs/e2e_dataset_sampler_with_FPN50_1x.yaml" FEATURE_DIR /gdata/megvii/coco/coco_featuremap_resnet50 OUTPUT_DIR ./ckpts
