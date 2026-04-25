#!/bin/bash --login
#SBATCH -p gpuA            # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate habitat

# EVAL
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 8 --model-name sp_pop_ppo_8net_invisible --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 8 --model-name sp_pop_ppo_8net_invisible --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --no-all-pairs --comm-field 100 --num-networks 8 --model-name sp_pop_ppo_8net_invisible --seed 3
