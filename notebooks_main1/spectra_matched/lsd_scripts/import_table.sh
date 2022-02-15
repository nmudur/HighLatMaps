#!/bin/bash
#SBATCH -J Exp2B
#SBATCH --account=finkbeiner_lab
#SBATCH -c 4
#SBATCH -t 0-01:00
#SBATCH -p test
#SBATCH --mem=40000
#SBATCH -o import_table_%j.o
#SBATCH -e import_table_%j.e

module load Anaconda/5.0.1-fasrc02
source activate fink_czucker

lsd-import fits sdss_dr17_specobj $LAB_PATH/catalogs/specObj-dr17.fits
