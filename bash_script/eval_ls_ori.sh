#!/bin/bash

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"


# EVAL
python -m scripts.torch_scoreg_layout.prepare_ls_traj_from_cpuckpt --no-all-pairs --num-networks 3 --ckpt-root checkpoints/pickup_high_v1 --model-name pop_ppo_3net_invisible --model-step 204800000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj_from_cpuckpt --no-all-pairs --num-networks 3 --ckpt-root checkpoints/pickup_high_v1 --model-name pop_ppo_3net_invisible --model-step 204800000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj_from_cpuckpt --no-all-pairs --num-networks 3 --ckpt-root checkpoints/pickup_high_v1 --model-name pop_ppo_3net_invisible --model-step 204800000 --seed 3


python -m scripts.torch_scoreg_layout.prepare_ls_traj_from_cpuckpt --no-all-pairs --num-networks 3 --ckpt-root checkpoints/pickup_high_v1 --model-name sp_pop_ppo_3net_invisible --model-step 204800000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj_from_cpuckpt --no-all-pairs --num-networks 3 --ckpt-root checkpoints/pickup_high_v1 --model-name sp_pop_ppo_3net_invisible --model-step 204800000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj_from_cpuckpt --no-all-pairs --num-networks 3 --ckpt-root checkpoints/pickup_high_v1 --model-name sp_pop_ppo_3net_invisible --model-step 204800000 --seed 3
