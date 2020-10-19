#!/bin/bash
list1=(1 2 3 4)
list2=(5 6 7 8)


for ((i = 0; i < 4; i++))
do
	export var1=${list1[i]}
	echo "Loop #: "$i "| list1 element: "${list1[i]} "| list2 element: "${list2[i]} "| variable: "$var1
done

for i in 512 1536 2048 128 1024
do
	PBSTASKS=`python3<<<"from math import floor; print((int(floor(${i}/48)) + (${i} % 48 > 0))*48)"`  # round up to nearest 48 as required by nci
	echo $PBSTASKS
done

export UW_DIM=3

if [ ${UW_DIM} -eq 3 ] ; then
	echo "CONDITION WORKS!"
    # cat spherical_sum.py
elif [ ${UW_DIM} -eq 2 ] ; then
	echo "CONDITION WORKS!"
    # cat annulus_sum_660.py
fi

multiple=`python3<<<"print(1.5*1024)"`
echo $multiple