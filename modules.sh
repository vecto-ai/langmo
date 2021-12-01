if [[ $(hostname) == *.abci.local ]]; then
    module load gcc/9.3.0
    module load cuda/11.1/11.1.1
    module load cudnn/8.1/8.1.1
    module load nccl/2.8/2.8.4-1
    module load openmpi/4.0.5
    module load python/3.8/3.8.7
fi
