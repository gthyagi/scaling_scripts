#!/bin/bash
export JOBS="1 2 4 8"
export UW_NAME="annulus"
export UW_ENABLE_IO="0"

export WALLTIME="02:00:00"
export ACCOUNT="n69"

export WEAK_SCALING_BASE=256
export UW_ORDER=1
export UW_SOL_TOLERANCE=1e-3
export UW_PENALTY=1e-3   # set to negative value to disable penalty 
export UW_SCRIPT="create_ann_swarm.py"
export UW_DIM=2
