#!/bin/bash

mkdir -p ./data/tokenized
python3 -m langmo.utils.preprocess roberta-base 256 ./data/corpora/sense ./data/tokenized/sense

#python3 -m langmo.utils.preprocess roberta-base 256 \
#    /mnt/storage/Data/NLP/corpora/original/openwebtext2 \
#    /mnt/scratch/tokenized/owt2
# python3 -m pstats token.profile
