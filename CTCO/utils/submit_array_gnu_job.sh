exp_name=$1
python_script=$2
echo $exp_name
config_base_path="/home/amirk96/scratch/CTCO_Experiments/${exp_name}/config_base.yaml"
num_params=$(yq -r .experiment.num_params $config_base_path)
num_runs=$(yq -r .experiment.num_runs $config_base_path)
num_jobs=$((num_params * num_runs))
echo $num_jobs
num_submit=$((num_jobs / 1000)) 
# for i in $(eval echo "{0..$num_submit}")
# do
#     echo $i
#     a=$(($i * 1000))
#     b=$(((i+1) * 1000))
#     b=$(( b < num_jobs ? b : num_jobs ))
#     echo "submiting $a-$((b-1))"
# done
echo $((num_params - 1))
sbatch --array=0-$((num_params - 1)) job_array_gnu.sh $exp_name $python_script