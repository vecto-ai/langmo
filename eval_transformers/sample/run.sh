#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=03:00:00
#$ -N NLI_sample
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# Create symbolic link to glue_data!!!
# ABCI: /groups2/gcb50300/data/NLP/datasets/glue_data
# matsulab: /work.fs/data/NLP/datasets/glue_data

source /etc/profile.d/modules.sh
source ../../modules.sh

python3 run_pl_glue.py \
	--output_dir=./out \
	--data_dir="glue_data/MNLI" \
	--do_train \
	--do_predict \
	--model_name_or_path=albert-base-v2 \
	--task=mnli \
	--max_seq_length=128 \
	--gpus=4

