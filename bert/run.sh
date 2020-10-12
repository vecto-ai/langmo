python3 train.py \
    --output_dir ./models/EsperBERTo-small-v1 \
    --model_type roberta \
    --mlm \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 16 \
    --evaluate_during_training \
    --seed 42 \
    --train_data_file /home/blackbird/Projects_heavy/NLP/langmo/data/wikitext-2/train.txt \
    --eval_data_file /home/blackbird/Projects_heavy/NLP/langmo/data/wikitext-2/test.txt

#    --config_name ./models/EsperBERTo-small \
#    --tokenizer_name ./models/EsperBERTo-small \
