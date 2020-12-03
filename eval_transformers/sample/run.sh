python3 run_pl_glue.py \
	--output_dir=./out \
	--data_dir="/work.fs/data/NLP/datasets/glue_data/MNLI" \
	--do_train \
	--do_predict \
	--model_name_or_path=albert-base-v2 \
	--task=mnli \
	--max_seq_length=128 \
	--gpus=4

