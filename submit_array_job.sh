exp_name=$1
python_script=$2
echo $exp_name
config_base_path="/home/amirk96/scratch/CTCO_Experiments/${exp_name}/config_base.yaml"
num_params=$(yq -r .experiment.num_params $config_base_path)
num_runs=$(yq -r .experiment.num_runs $config_base_path)
num_jobs=$((num_params * num_runs))
echo $num_jobs
num_submit=$((num_jobs / 1000)) 
for i in $(eval echo "{0..$num_submit}")
do
    echo $i
    a=$(($i * 1000))
    b=$(((i+1) * 1000))
    b=$(( b < num_jobs ? b : num_jobs ))
    echo "submiting $a-$((b-1))"
    if [[ $i -gt 0 ]]
    then
    echo "gt"
    prev_JOBID=$(sbatch --dependency=afterany:$prev_JOBID --array=0-$(((b-1) % 1000)) job_array.sh $exp_name $python_script $i)
    else
    echo $(((b-1) % 1000))
    prev_JOBID=$(sbatch --array=0-$(((b-1) % 1000)) job_array.sh $exp_name $python_script 0)
    fi
    
done
