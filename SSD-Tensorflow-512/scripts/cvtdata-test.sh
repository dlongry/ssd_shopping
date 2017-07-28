#!/bin/sh

DATASET_DIR=`pwd`/../../VOCdevkit-test/VOC2007/
OUTPUT_DIR=`pwd`/../../VOC-tfrecords/
python ../tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=voc_2007_test \
    --output_dir=${OUTPUT_DIR}
