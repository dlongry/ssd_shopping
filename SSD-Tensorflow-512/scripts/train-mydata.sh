#!/bin/sh

DATASET_DIR=../../VOC-mydata-tfrecords/
TRAIN_DIR=../logs/
CHECKPOINT_PATH=../checkpoints/vgg_16.ckpt
python ../train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=pascalvoc_mydata \
    --dataset_split_name=train \
    --model_name=ssd_512_vgg \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_exclude_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --trainable_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=5
