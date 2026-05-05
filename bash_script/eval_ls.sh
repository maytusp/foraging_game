#!/bin/bash

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"


# EVAL
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --seed 3


# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --seed 3


python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --seed 3


# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --seed 3



python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 100 --model-name pop_ppo_100net_invisible --model-step 2483200000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 100 --model-name pop_ppo_100net_invisible --model-step 2483200000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 100 --model-name pop_ppo_100net_invisible --model-step 2483200000 --seed 3


# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 100 --model-name sp_pop_ppo_100net_invisible --model-step 1996800000 --seed 1
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 100 --model-name sp_pop_ppo_100net_invisible --model-step 1996800000 --seed 2
# python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --num-networks 100 --model-name sp_pop_ppo_100net_invisible --model-step 1996800000 --seed 3



# # TRAIN