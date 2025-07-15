#!/bin/bash -l
#
#SBATCH -J CalcSpeed
#SBATCH -p normal
#SBATCH -t 2-00:00:00
#SBATCH -n 32
#SBATCH -o CalculateSpeed_areaave_weighted-log.%j.o   
#SBATCH -e CalculateSpeed_areaave_weighted-log.%j.e

module load miniconda
eval "$(conda shell.bash hook)"  # this makes sure that conda works in the batch environment 

cd /nethome/ruhs0001/IMMERSE_waves/develop-lorenz/code/
conda activate parcels

python3 CalculateSpeed_areaave_weighted.py
