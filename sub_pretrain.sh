if git config --get user.email > /dev/null; then
    EMAILARGS="-m bea -M $(git config --get user.email)"
else
    echo "$0 WARNING: git email not set!"
fi

#if [[ $SGE_CLUSTER_NAME = "t3" ]]; then
    # TSUBAME 3.0
#    qsub -g tgc-ebdcrest run.sh
#else
    # probably TSUBAME-KFC/DL
    # sbatch run_kfc.sh
    # abci
    qsub -g gcb50300 $EMAILARGS run_pretrain.sh
    # qsub -ar 5476 -g gad50785 $EMAILARGS run_pretrain.sh
#fi
