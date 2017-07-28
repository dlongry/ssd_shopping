#!/bin/sh

OUTPUT_NAME="voc_mydata_train"

[ "$1" = 1 ] && {
OUTPUT_NAME="voc_mydata_test"
}

echo $OUTPUT_NAME

DATASET_DIR=../../ourdata/
OUTPUT_DIR=../../VOC-mydata-tfrecords/
python ../tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${DATASET_DIR} \
    --output_name=${OUTPUT_NAME} \
    --output_dir=${OUTPUT_DIR}
