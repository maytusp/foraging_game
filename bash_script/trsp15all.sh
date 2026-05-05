#!/bin/bash

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"



python -m scripts.torch_scoreg_layout.train_ori --comm_field 100 --image_size 3 --no-agent-visible --num_networks 15 --self_play_option --seed 1 --total-timesteps 400000000
python -m scripts.torch_scoreg_layout.train_ori --comm_field 100 --image_size 3 --no-agent-visible --num_networks 15 --self_play_option --seed 2 --total-timesteps 400000000
python -m scripts.torch_scoreg_layout.train_ori --comm_field 100 --image_size 3 --no-agent-visible --num_networks 15 --self_play_option --seed 3 --total-timesteps 400000000
