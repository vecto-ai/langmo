langmo
######

The library for distributed pretraining and finetuning of language models.

Supported features:

- vanilla pre-training of BERT-like models
- distributed training on multi-node/multi-GPU systems
- benchmarking/finetuning the following tasks
    - all GLUE
    - MNLI  + additional validation on HANS
    - more coming soon
- using siamese architectures for finutuning


Pretraining
-----------

Pretraining a model::

    mpirun -np N python -m langmo.pretraining config.yaml

langmo saves 2 types of snapshots: in pytorch_ligning format 

To resume crashed/aborted pretraining session:

    mpirun -np N python -m langmo.pretraining.resume path_to_run


Finetuning/Evaluation
---------------------

Finetuning on one of the GLUE tasks::

    mpirun -np N python -m langmo.benchmarks.GLUE config.yaml glue_task

supported tasks: **cola, rte, stsb, mnli, mnli-mm, mrpc, sst2, qqp, qnli**

NLI task has additional special implentation which supports validation on adversarial HANS dataset,
as well as additional staticics for each label/heuristic.

To perfrorm fibetuning on NLI run as::

    mpirun -np N python -m langmo.benchmarks.NLI config.yaml


Finetuning on extractive question-answering tasks::

    mpirun -np N python -m langmo.benchmarks.QA config.yaml qa_task

supported tasks: **squad, squad_v2**

example config file:

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


Automatic evaluation
--------------------

langmo supports automatic scheduling of evaluation runs for a model saved in a given location, or for all snapshots found int /snapshots folder.
To configure langmo the user has to create the following file:

./configs/langmo.yaml with entry "submit_command" correspoding to a job submission command of a given cluster. If the file is not present, the jobs will not be submitted to the job queue, but executed immediately one by one on the same node.

./configs/auto_finetune.inc - the content of this file will be copied to the beginning of the job scripts. Place here directive for e.g. slurm job scheduler such as 
which resource group to use, how many nodes to allocate, time limit etc. Set up all necessary environment variables, particulalry NUM_GPUS_PER_NODE and
PL_TORCH_DISTRIBUTED_BACKED (MPI, NCCL or GLOO). Finally add mpirun command with necessay option and end the file with new line.
Command to invoke langmo in the right way will be added automatically.

./configs/auto_finetune.yaml - any parameters such as batch size etc to owerride the defaults in a fine-tuning run.

To schedule evaluation jobs run from the login node::

    python -m langmo.benchmarks path_to_model task_name

the results will be saved in the eval/task_name/run_name/ subfolder in the same folder the model is saved.

Fugaku notes
------------

Add these lines before the :code:`return` of :code:`_compare_version`
statement of :code:`pytorch_lightning/utilities/imports.py`.::

    if str(pkg_version).startswith(version):
        return True

This :code:`sed` command should do the trick::

    sed -i -e '/pkg_version = Version(pkg_version.base_version/a\    if str(pkg_version).startswith(version):\n\        return True' \
      ~/.local/lib/python3.8/site-packages/pytorch_lightning/utilities/imports.py
