#!/bin/sh

#parallel.sbatch
#SBATCH --nodes=5 
#SBATCH --ntasks=240 
#SBATCH -t 0-16:00
#SBATCH -p itc_cluster,shared # Partition
#SBATCH --mem=0
#SBATCH -J parallel_compute_posteriors # A single job name for the array
#SBATCH --account=finkbeiner_lab

#SBATCH --mail-user=nmudur@cfa.harvard.edu
#SBATCH --mail-type=END

#SBATCH --signal=USR1@4
#SBATCH --signal=USR1@2

module load parallel

NTASKS="$(($SLURM_NTASKS-5))"
MEM=4000
dirname=/n/holylfs05/LABS/finkbeiner_lab/Lab/nmudur/bayestar_edr3/stout/fullsky/postlogs/south/
runlogfile=$dirname/"runtasks_south.log"
srunlogfile=$dirname/"log_${SLURM_JOB_ID}_"
srun="srun --exclusive -n1 -N1 --mem-per-cpu=${MEM}"
parallel="parallel --delay .2 -j $NTASKS --joblog $runlogfile --resume-failed"

$parallel "$srun bash fullsky_south.sh {1} 1> $srunlogfile{1}.log 2> $srunlogfile{1}.err" ::: {0..1283}
