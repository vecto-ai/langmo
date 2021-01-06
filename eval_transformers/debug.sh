#!/bin/bash

JOB_NAME=$(grep -e '#\$ -N' run.sh | sed -e 's/.*-N //')

git pull \
&& rm -rf $JOB_NAME.o* \
&& qsub -g gcb50300 run.sh \
&& until ls $JOB_NAME.o* 2>/dev/null; do sleep 1; echo -n .; done \
&& tail -f $JOB_NAME.o*
