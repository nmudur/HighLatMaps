#!/bin/bash
#SBATCH -J Exp2B
#SBATCH --account=finkbeiner_lab
#SBATCH -c 4
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH --mem=4000
#SBATCH -o logs/test_%j.o
#SBATCH -e logs/test_%j.e

module load Anaconda/5.0.1-fasrc02
source activate fink_czucker

echo $LAB_PATH
echo $LSD_DB
