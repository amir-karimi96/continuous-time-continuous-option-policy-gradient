#!/bin/bash
#SBATCH --mail-user=amir.karimi96@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-ashique
#SBATCH --time=00:59:00
#SBATCH --mem=2000M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=/home/amirk96/scratch/outputs/%x-%j.out
echo 'Hello, world!'
exp_name=$1
python_script=$2

source ~/projects/def-ashique/amirk96/CTCO_venv/bin/activate

config_base_path="/home/amirk96/scratch/CTCO_Experiments/${exp_name}/config_base.yaml"
num_runs=$(yq -r .experiment.num_runs $config_base_path)
run_ID=$(((SLURM_ARRAY_TASK_ID) % num_runs))
config_ID=$(((SLURM_ARRAY_TASK_ID) / num_runs))

config_path="/home/amirk96/scratch/CTCO_Experiments/${exp_name}/configs/config_${config_ID}.yaml"
result_path="/home/amirk96/scratch/CTCO_Experiments/${exp_name}/results"
OMP_NUM_THREADS=1 python3 $python_script --config=$config_path --ID=$run_ID --result_path=$result_path

