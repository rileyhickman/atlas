#!/bin/bash
#SBATCH --account=rrg-aspuru
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem=1000M
#SBATCH --time=00:00:30
#SBATCH -o output333.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malcolm.sim@mail.utoronto.ca
#SBATCH --job-name test_job2

source atlas-env/bin/activate
pip freeze --local > requirements.txt

python3 test.py
