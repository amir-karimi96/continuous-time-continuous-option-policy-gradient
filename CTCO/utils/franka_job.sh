#!/bin/bash

echo 'Hello, world!'
exp_name=$1
python_script=$2

config_ID="config_1"

config_path="/home/franka/project/CTCO_scratch/CTCO_Experiments/${exp_name}/configs/${config_ID}.yaml"
result_path="/home/franka/project/CTCO_scratch/CTCO_Experiments/${exp_name}/results"

for i in {910..910}
do
   echo "$i"
   OMP_NUM_THREADS=1 python3 $python_script --config=$config_path --ID=$i --result_path=$result_path
done

# OMP_NUM_THREADS=1 python3 $python_script --config=$config_path --ID=$run_ID --result_path=$result_path

