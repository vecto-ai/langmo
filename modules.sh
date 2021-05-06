if [[ $(hostname) == *.abci.local ]]; then
    module load gcc/9.3.0
    module load cuda/11.2/11.2.2
    # module load cuda/10.2/10.2.89
    #module load cudnn/7.6/7.6.5
    module load cudnn/8.1/8.1.1
    # module load nccl/2.5/2.5.6-1
    module load nccl/2.8/2.8.4-1
    # module load openmpi/2.1.6
    # module load intel-mpi/2019.5
    module load openmpi/4.0.5
    module load python/3.8/3.8.7
    #module load python/3.8/3.8.2
    #module load python/3.6/3.6.5

    #source /home/aca10027xu/opt/extra/spack/share/spack/setup-env.sh
    #source /home/aca10027xu/opt/scripts/env/my_env.sh
    #source /home/aca10027xu/apps.sh
fi
