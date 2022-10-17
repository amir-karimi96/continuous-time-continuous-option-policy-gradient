import yaml
import os
from pathlib import Path
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_directory',default='None',type=str, help='experiments folder name')
parser.add_argument('--exp_name',default='None',type=str, help='exp folder name')

args = parser.parse_args()

exps_dir = args.experiments_directory
exp_name = args.exp_name


# a base config file should exist
config_base_path = "{}/{}/config_base.yaml".format(exps_dir ,exp_name)
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
Path("{}/{}/configs".format(exps_dir ,exp_name)).mkdir(parents=True, exist_ok=True)
for i in range(len(configs)):
    Path("{}/{}/results/config_{}/plots".format(exps_dir ,exp_name,i)).mkdir(parents=True, exist_ok=True)
    Path("{}/{}/results/config_{}/data".format(exps_dir ,exp_name,i)).mkdir(parents=True, exist_ok=True)
    Path("{}/{}/results/config_{}/model".format(exps_dir ,exp_name,i)).mkdir(parents=True, exist_ok=True)


## write configs
documents['experiment']['num_params']=len(P)
with open(config_base_path, 'w') as file:
    documents = yaml.safe_dump(documents, file)

for ID,c in enumerate(configs):
    config_path = os.path.join('{}/'.format(exps_dir),os.path.dirname(__file__), '{}/configs/config_{}.yaml'.format(exp_name, ID))
    # print(config_path)

    with open(config_path, 'w') as file:
        documents = yaml.dump(c, file)