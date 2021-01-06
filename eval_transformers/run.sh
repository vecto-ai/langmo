#!/bin/bash
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=00:59:00
#$ -N NLI_run_glue
#$ -j y
#$ -o $JOB_NAME.o$JOB_ID

# Create symbolic link to glue_data!!!
# ABCI: /groups2/gcb50300/data/NLP/datasets/glue_data
# matsulab: /work.fs/data/NLP/datasets/glue_data

#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export DATA_DIR=/mnt/storage/Data/NLP/importance_sampling/subsample_rand/seed_42/6
export DATA_DIR=/groups2/gcb50300/chinese_room/subsample_rand/seed_42/6
export TASK_NAME=MNLI
MODEL_NAME=albert-base-v2
#bert_base_cased
#MODEL_NAME=distilbert-base-cased

if [[ $(hostname) == *.abci.local ]]; then
    source /etc/profile.d/modules.sh
    source ../modules.sh
fi

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` 

echo $DATE_WITH_TIME

WANDB_PROJECT=$JOB_NAME
echo "wandb project: $WANDB_PROJECT"

python3 run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_steps 1000 \
  --evaluation_strategy "steps" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --output_dir DL_outs/NLP/$TASK_NAME/$MODEL_NAME/$DATE_WITH_TIME \
  --logging_dir DL_outs/NLP/$TASK_NAME/$MODEL_NAME/$DATE_WITH_TIME/log \
  --overwrite_output_dir \
  --logging_steps 1000 \
  --save_steps -1 \
