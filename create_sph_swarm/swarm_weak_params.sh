#!/bin/bash
export JOB_NCPUS=(2 16 128 1024)
export UW_NAME="spherical_swarm"
export UW_ENABLE_IO="0"

export WALLTIME="02:00:00"
export ACCOUNT="n69"

export RES_RAD=(32 64 128 256)
export RES_LON=(48 96 192 384)
export RES_LAT=(64 128 256 512)
export WEAK_SCALING_BASE="32_48_64"
export UW_ORDER=1
export UW_SOL_TOLERANCE=1e-3
export UW_PENALTY=1e-3   # set to negative value to disable penalty 
export UW_SCRIPT="create_swarm_sum.py"
export UW_DIM=3
