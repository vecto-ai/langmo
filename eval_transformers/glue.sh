#export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
#export DATA_DIR=/mnt/storage/Data/NLP/importance_sampling/subsample_rand/seed_42/6
export DATA_DIR=/groups2/gcb50300/chinese_room/subsample_rand/seed_42/6
export TASK_NAME=MNLI
MODEL_NAME=albert-base-v2
#bert_base_cased
#MODEL_NAME=distilbert-base-cased

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"` 
  
echo $DATE_WITH_TIME

python3 run_glue.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 64 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --output_dir /groups2/gcb50300/DL_outs/NLP/$TASK_NAME/$MODEL_NAME/$DATE_WITH_TIME \
  --logging_dir /groups2/gcb50300/DL_outs/NLP/$TASK_NAME/$MODEL_NAME/$DATE_WITH_TIME/log \
  --overwrite_output_dir \
  --logging_steps 200 \
  --save_steps -1 \
