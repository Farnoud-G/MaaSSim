#!/bin/bash
#SBATCH --job-name=massim-test
#SBATCH --qos=big
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Go to the folder you wanna run jupyter in
cd /home/ghasemi/MaaSSim/docs/RL_EXP/

echo "run_id: $1";
echo "lr: $2";
echo "stp: $3";

# Start the notebook
python -m pip install -r requirements.txt
python -u main_run.py $1 $2 $3 >> mout_$1.txt
