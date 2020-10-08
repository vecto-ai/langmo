from transformers import AutoModelForSequenceClassification, AutoConfig

config = AutoConfig.from_pretrained(
    "albert-base-v2", num_labels=3)
# finetuning_task=data_args.task_name,
# cache_dir=model_args.cache_dir)

net = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", config=config)
