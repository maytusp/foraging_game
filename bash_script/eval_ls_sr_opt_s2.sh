#!/bin/bash --login
#SBATCH -p gpuL             # A100 GPUs
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
# For WS Cannot Do this. Need --graph-structure....
# # LS
python -m scripts.torch_scoreg_layout.prepare_ls_struct --no-all-pairs --graph-structure opt_pairs_100 --num-networks 100 --model-name opt_ppo_100net_invisible --model-step 1200000000 --seed 2

# SR 
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 100 --model-name opt_ppo_100net_invisible --model-step 1200000000 --seed 2


