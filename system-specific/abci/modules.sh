if [[ $(hostname) == *.abci.local ]]; then
    source /etc/profile.d/modules.sh
    module load gcc/11.2.0
    module load openmpi/4.1.3
    module load cuda/11.5/11.5.2
    module load cudnn/8.3/8.3.3
    module load nccl/2.11/2.11.4-1
    module load python/3.10/3.10.4
fi
