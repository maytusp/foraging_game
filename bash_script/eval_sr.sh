#!/bin/bash

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"


python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name ring_ppo_100net_invisible --model-step 1996800000 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name ring_ppo_100net_invisible --model-step 1996800000 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name ring_ppo_100net_invisible --model-step 1996800000 --seed 3


# python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name sp_ring_ppo_100net_invisible --model-step 1996800000 --seed 1
# python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name sp_ring_ppo_100net_invisible --model-step 1996800000 --seed 2
# python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name sp_ring_ppo_100net_invisible --model-step 1996800000 --seed 3

#  CUDA_VISIBLE_DEVICES=0 bash eval_sr.sh