#! /usr/bin/env bash
set -e
cd "$(dirname "$0")"
. util.sh

# https://www.kaggle.com/models/google/yamnet/tfLite/tflite

yamnet_dir="$(config yamnet_dir)"
yamnet_model="$(config yamnet_model)"
yamnet_class_map="$(config yamnet_class_map)"

(
    mkdir -p "$yamnet_dir"
    cd "$yamnet_dir"

    if [[ ! -f "$yamnet_model" ]]; then
        wget -O yamnet.tflite.tar.gz https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/tflite/1/download
        tar xzf yamnet.tflite.tar.gz
        rm yamnet.tflite.tar.gz
        mv 1.tflite "$yamnet_model"
    fi

    if [[ ! -f "$yamnet_class_map" ]]; then
        wget -O "$yamnet_class_map" https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/audioset/yamnet/yamnet_class_map.csv
    fi
)
