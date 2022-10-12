#!/bin/bash
LD_PRELOAD=./wrapper_libs/libnccl.so N_WARMUP=10 N_STEP=10 FLOPS_NAME="${MODEL_NAME}" nsys profile --force-overwrite true -o ./nsys_output/${MODEL_NAME} python real_train.py

if [ $? -eq 0 ] 
then
    echo "Success Run Program."
else
    echo "
XXXXXXXXXXX Failed Experiment XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    "
    exit 1
fi

echo "
----------- Finish Experiment --------------------------------------------------------------------
"