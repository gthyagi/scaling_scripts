#!/bin/bash
export JOBS="1 2 4 8"
export UW_NAME="thyagi2d_test_take1"
export UW_ENABLE_IO="0"

export WALLTIME="00:01:00"
export ACCOUNT="n69"

export WEAK_SCALING_BASE=128
export UW_ORDER=1
export UW_SOL_TOLERANCE=1e-3
export UW_PENALTY=1e-3   # set to negative value to disable penalty 
export UW_SCRIPT="1024_512_sum_660.py"
export UW_DIM=2