#!/usr/bin/env python
#SBATCH --ntasks=48
#SBATCH -t 0-00:30
#SBATCH -p test
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH -J NGCSmall
#SBATCH --account=finkbeiner_lab

import os
os.system("module load Anaconda3/2020.11")
os.system("module load parallel")

NTASKS = int(os.environ["SLURM_NTASKS"]) - 1
dirname = os.environ["TMPDIR"]
logdir = os.path.join(dirname, 'logs')+'/'
parallel = "parallel --delay .2 -j %d --joblog %sruntasks_%s.log --resume-failed" % (NTASKS, dirname, os.environ["SLURM_JOB_ID"])
srun = "srun --exclusive -N1 -n1 --mem-per-cpu="+os.environ["MEM"]
condapy = "/n/home02/nmudur/.conda/envs/pytorch_env2/bin/python"
pyfile = "/n/holylfs05/LABS/finkbeiner_lab/Everyone/highlat/methods_code_Nresol/do_recon_tilewise.py"
print("Here")
os.system('%s -a %s "%s %s %s {} %s >> %ssrun_%s_{}.log"' % (parallel, os.environ["TILEFILE"], srun, condapy, pyfile, dirname, logdir, os.environ["SLURM_JOB_ID"]))