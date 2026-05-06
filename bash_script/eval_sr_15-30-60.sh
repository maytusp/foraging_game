#!/bin/bash

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"




# LS SR 15 net
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 15 --model-name pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 15 --model-name pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 15 --model-name pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 3

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 15 --model-name sp_pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 15 --model-name sp_pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 15 --model-name sp_pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 3


python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 15 --model-name pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 15 --model-name pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 15 --model-name pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 3

python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 15 --model-name sp_pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 15 --model-name sp_pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 15 --model-name sp_pop_ppo_15net_invisible --model-step 399989760 --ckpt-root checkpoints/torch_scoreg_layout2 --seed 3


# LS SR 30 net

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 3

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 3


python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 30 --model-name pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 3

python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 30 --model-name sp_pop_ppo_30net_invisible --model-step 1499996160 --ckpt-root checkpoints/scoreg --seed 3



# LS SR 60 net

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --ckpt-root checkpoints/scoreg --seed 3

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --ckpt-root checkpoints/scoreg --seed 3


python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 60 --model-name pop_ppo_60net_invisible --model-step 1799992320 --ckpt-root checkpoints/scoreg --seed 3

python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --ckpt-root checkpoints/scoreg --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --ckpt-root checkpoints/scoreg --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 60 --model-name sp_pop_ppo_60net_invisible --model-step 1561600000 --ckpt-root checkpoints/scoreg --seed 3

