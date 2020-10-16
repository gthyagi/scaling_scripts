#!/bin/bash
module purge
RUN_MODS='pbs openmpi/4.0.2 hdf5/1.10.5p python3/3.7.4'
module load scons/3.1.1 $RUN_MODS
export PYTHONPATH=/home/565/tg7098/usr_installed_modules/uw_regional/underworld:/home/565/tg7098/usr_installed_modules/uw_regional/underworld/libUnderworld/build/lib:/home/565/tg7098/usr_installed_modules/uw_regional/underworld/glucifer:
export LD_PRELOAD=:/apps/openmpi-mofed4.7-pbs19.2/4.0.2/lib/libmpi_usempif08_GNU.so.40
export LD_PRELOAD=:/apps/openmpi-mofed4.7-pbs19.2/4.0.2/lib/libmpi_usempi_ignore_tkr_GNU.so.40
export LD_PRELOAD=:/apps/openmpi-mofed4.7-pbs19.2/4.0.2/lib/libmpi_cxx.so.40
echo "Loaded all modules"
echo "Executing python script"

env
cat 1024_512_sum_660.py

echo "-------------------------------------------------------------------------------------------"
export TIME_LAUNCH_MPI=`date +%s%N | cut -b1-13`
mpirun -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_SCRIPT}"

