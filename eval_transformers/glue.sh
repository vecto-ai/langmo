#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export GLUE_DIR=/mnt/storage/data/NLP/datasets/glue_data/
export GLUE_DIR=/work.fs/data/NLP/datasets/glue_data
export TASK_NAME=MNLI


# TODO: add timestamp
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` 
  
echo $DATE_WITH_TIME

python3 run_glue.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --output_dir /work.fs/alex/DL_outs/NLP/$TASK_NAME/$DATE_WITH_TIME \
  --overwrite_output_dir \
  --logging_steps 200 \
  --save_steps -1 \
  --num_cores=8 \
  --only_log_master