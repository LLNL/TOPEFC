#!/bin/tcsh
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A eng

##### Launch parallel job using srun
srun -N1 -n4 python3 ./supercapacitor.py --Nx 64 --proj 4.0 > output.txt
echo 'Done'
