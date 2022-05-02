from math import exp
from numpy.core import numeric
import yaml
import os
from pathlib import Path
from itertools import product
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',default='None',type=str, help='exp folder name')
args = parser.parse_args()

exp_name = args.exp_name


# a base config file should exist
config_base_path = "/home/amirk96/scratch/CTCO_Experiments/{}/config_base.yaml".format(exp_name)
## read config_base
with open(config_base_path,'r') as file:
    documents = yaml.full_load(file)
    prod_list = []
    item_list = []
    for item, doc in documents['experiment']['params'].items():
        print(item, ":", doc)
        prod_list.append(doc)
        item_list.append(item)



P = list(product( *prod_list ))
params = [dict(zip(item_list, p)) for p in P]
configs = [{'exp_name':exp_name, 'param': k, 'param_ID': i} for i,k in enumerate(params)]

# make directories
Path("/home/amirk96/scratch/CTCO_Experiments/{}/configs".format(exp_name)).mkdir(parents=True, exist_ok=True)
for i in range(len(configs)):
    Path("/home/amirk96/scratch/CTCO_Experiments/{}/results/config_{}/plots".format(exp_name,i)).mkdir(parents=True, exist_ok=True)
    Path("/home/amirk96/scratch/CTCO_Experiments/{}/results/config_{}/data".format(exp_name,i)).mkdir(parents=True, exist_ok=True)


## write configs
documents['experiment']['num_params']=len(P)
with open(config_base_path, 'w') as file:
    documents = yaml.safe_dump(documents, file)

for ID,c in enumerate(configs):
    config_path = os.path.join('/home/amirk96/scratch/CTCO_Experiments/',os.path.dirname(__file__), '{}/configs/config_{}.yaml'.format(exp_name, ID))
    # print(config_path)

    with open(config_path, 'w') as file:
        documents = yaml.dump(c, file)