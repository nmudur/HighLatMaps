#!/bin/sh

module load Anaconda/5.0.1-fasrc02
source activate pytorch_env2 

gridfile="$1"
finalfile="$2"
deletecmd="$3"

echo "Grid File: ${gridfile}"
echo "Output File: ${finalfile}"
echo "Delete Command: ${deletecmd}"

python /n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/combine_inpout.py ${gridfile} ${finalfile} ${deletecmd} 
