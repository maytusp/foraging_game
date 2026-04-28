#!/bin/bash --login
#SBATCH -p gpuL              # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 4-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate habitat

python -m scripts.torch_scoreg_layout.train_ori --comm_field 100 --image_size 3 --no-agent-visible --num_networks 30 --no-self_play_option --seed 1 --total-timesteps 1500000000
python -m scripts.torch_scoreg_layout.train_ori --comm_field 100 --image_size 3 --no-agent-visible --num_networks 30 --no-self_play_option --seed 2 --total-timesteps 1500000000
python -m scripts.torch_scoreg_layout.train_ori --comm_field 100 --image_size 3 --no-agent-visible --num_networks 30 --no-self_play_option --seed 3 --total-timesteps 1500000000
