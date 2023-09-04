#!/usr/bin/bash
pjsub                             \
    --sparam wait-time=60         \
    --no-check-directory          \
    ./fu_pretrain_mpi.sh
