#!/bin/bash
#SBATCH -J dr17_test 
#SBATCH --account=finkbeiner_lab
#SBATCH -c 4
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH --mem=8000
#SBATCH -o logs/lsd_chunk_%j.o
#SBATCH -e logs/lsd_chunk_%j.e

module load Anaconda/5.0.1-fasrc02
source activate LSD_env2
export OPTIONS="select lsd_id, PLUG_RA, PLUG_DEC, OBJID, ZWARNING, CHI68P, SN_MEDIAN_ALL, SN_MEDIAN, RCHI2, CLASS, SUBCLASS, Z, Z_ERR from sdss_dr17_specobj"
lsd-query --format=fits --output=test_table_r5.fits --bounds='beam(0, 90, 5, coordsys="gal")' "$OPTIONS"
