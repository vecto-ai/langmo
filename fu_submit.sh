#!/usr/bin/bash
pjsub                             \
    -x PJM_LLIO_GFSCACHE=/vol0004 \
    --sparam wait-time=60         \
    --no-check-directory          \
    ./fu_pretrain_mpi.sh

