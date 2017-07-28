#!/bin/sh

DATASET_DIR=../../VOC-mydata-tfrecords/
EVAL_DIR=../logs/
CHECKPOINT_PATH=../logs/model.ckpt-103343
python ../eval_ssd_network.py \
    --eval_dir=${EVAL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_mydata \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --nms_threshold=0.05 \
    --batch_size=1
