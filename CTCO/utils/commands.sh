python3 run_sac.py --r=/home/amirk96/scratch/CTCO_Experiments/SAC_half_cheetah/results --c=/home/amirk96/scratch/CTCO_Experiments/SAC_half_cheetah/configs/config_3.yaml --ID=100
bash submit_array_gnu_job.sh SAC_pendulum run_sac.py
python3 plotter_spec.py --exp_name=test_pendulum --experiments_directory=/home/amirk96/scratch/CTCO_Experiments --w=2
python3 exp_init.py --exp_name=f_SAC_mountain_car --experiments_directory=/home/amirk96/scratch/CTCO_Experiments