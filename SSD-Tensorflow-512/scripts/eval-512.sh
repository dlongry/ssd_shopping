#!/bin/sh

DATASET_DIR=../../VOC-tfrecords/
EVAL_DIR=../logs/
CHECKPOINT_PATH=../checkpoints/model.ckpt-73389
python ../eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_2007 \
    --dataset_split_name=test \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --batch_size=1
