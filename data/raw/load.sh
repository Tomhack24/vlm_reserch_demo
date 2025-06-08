#!/bin/bash

DATA_DIR="./VQAv2_val"

mkdir -p "${DATA_DIR}"

curl -L -o "${DATA_DIR}/v2_Questions_Val_mscoco.zip" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
curl -L -o "${DATA_DIR}/v2_Annotations_Val_mscoco.zip" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

unzip -o "${DATA_DIR}/v2_Questions_Val_mscoco.zip" -d "${DATA_DIR}"
unzip -o "${DATA_DIR}/v2_Annotations_Val_mscoco.zip" -d "${DATA_DIR}"
