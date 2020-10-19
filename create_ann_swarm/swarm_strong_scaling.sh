#!/bin/bash
export STRONG_SCALING_BASE=1024
export UW_NAME="annulus"
export UW_ENABLE_IO="0"
export WALLTIME="04:00:00"
export ACCOUNT="n69"
export UW_ORDER=1
export UW_SOL_TOLERANCE=1e-3
export UW_PENALTY=1e-3   # set to negative value to disable penalty 
export UW_SCRIPT="create_ann_swarm.py"
export UW_DIM=2
export NAME="Results_strong_scaling_${UW_NAME}_DIM_${UW_DIM}_BASE_${STRONG_SCALING_BASE}_ORDER_${UW_ORDER}_TOL_${UW_SOL_TOLERANCE}_PENALTY_${UW_PENALTY}_IO_${UW_ENABLE_IO}"

## find the BATCH environment ##
#################################
if qstat --version > /dev/null ; then
   BATCH_SYS="PBS"
elif squeue --version > /dev/null ; then
   BATCH_SYS="SLURM"
else
   echo "Can't workout batch system"
   exit 1
fi

echo "Batch system is $BATCH_SYS"
#################################
mkdir -p ${NAME}
cp *.sh ${NAME}
cp *.py ${NAME}
cd ${NAME}

for i in 1 2 4 8 12 
do
   export UW_RESOLUTION=${STRONG_SCALING_BASE}
   export NTASKS="$((${i}*${i}))"
   export EXPORTVARS="UW_RESOLUTION,NTASKS,UW_ENABLE_IO,UW_ORDER,UW_DIM,UW_SOL_TOLERANCE,UW_PENALTY,UW_SCRIPT"
   if [ $BATCH_SYS == "PBS" ] ; then
      export QUEUE="normal" # normal or express

      # memory requirement guess: 3GB * nprocs
      MEMORY="$((4*${NTASKS}))GB"
      if [ ${NTASKS} -le 48 ] ; then
         PBSTASKS=`python3<<<"print(${NTASKS})"`
	 export PBSTASKS
      else
         PBSTASKS=`python3<<<"from math import floor; print((int(floor(${NTASKS}/48)) + (${NTASKS} % 48 > 0))*48)"`  # round up to nearest 48 as required by nci
	 export PBSTASKS
      fi
      echo "Python job cpus: "${NTASKS} "| PBS jobs cpus: "${PBSTASKS}
      export EXPORTVARS="UW_RESOLUTION,NTASKS,UW_ENABLE_IO,UW_ORDER,UW_DIM,UW_SOL_TOLERANCE,UW_PENALTY,UW_SCRIPT,PBSTASKS"
      # -V to pass all env vars to PBS (raijin/nci)
      CMD="qsub -v ${EXPORTVARS} -N ${NAME} -l ncpus=${PBSTASKS},mem=${MEMORY},walltime=${WALLTIME},wd -P ${ACCOUNT} -q ${QUEUE} gadi_baremetal_go.sh"
      echo ${CMD}
      ${CMD}
   else
      export IMAGE=/group/m18/singularity/underworld/underworld2_2.10.0b_rc.sif
      #export IMAGE=/group/m18/singularity/underworld/underworld2_v29.sif
      export QUEUE="workq" # workq or debugq
      export OUTNAME="Res_"${UW_RESOLUTION}"_Nproc_"${NTASKS}"_JobID_"%j".out"

      CMD="sbatch --export=IMAGE,${EXPORTVARS} --job-name=${NAME} --output=${OUTNAME} --ntasks=${NTASKS} --time=${WALLTIME} --account=${ACCOUNT} --partition=${QUEUE} magnus_container_go.sh"
      echo ${CMD}
      ${CMD}
   fi
done

