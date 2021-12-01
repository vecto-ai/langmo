langmo
######

The library for distributed pretraining and finetuning of language models.

Supported features:

- vanilla pre-training on BERT-like models
- finetuning the following datasets
    - MNLI  + additional validation on HANS
    - SST
    - more coming soon
- using siamese architectures for finutuning


to perfrorm fibetuning on a task, e.g. NLI run as::

    horovodrun -np N python -m langmo.benchmarks.NLI config.yaml

example cofig file:

::

    model_name: "roberta-base"
    batch_size: 32
    cnt_epochs: 4
    path_results: ./logs
    max_lr: 0.0005
    siamese: true
    freeze_encoder: false
    encoder_wrapper: pooler
    shuffle: true


Fugaku notes
------------

Add these lines before the :code:`return` of :code:`_compare_version`
statement of :code:`pytorch_lightning/utilities/imports.py`.::

    if str(pkg_version).startswith(version):
        return True

