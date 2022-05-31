#!/bin/bash

mkdir -p ./data/tokenized
# python3 -m langmo.utils.preprocess roberta-base 128 ./data/sense ./data/tokenized/sense
python3 -m langmo.utils.preprocess roberta-base 30 ./data/sample ./data/tokenized/sample
