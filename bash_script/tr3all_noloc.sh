#!/bin/bash --login
#SBATCH -p gpuL             # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 3-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate habitat

# No-localisation ablation (PPOLSTMCommAgentNoLoc): population of 3, 400M steps per seed.
# fc social network = population sampling (any pair), matching the pop_ppo_3net_invisible baselines.

python -m scripts.torch_scoreg_layout.train_struct_noloc --social-network fc --comm_field 100 --no-agent-visible --num_networks 3 --seed 1 --no-self-play-option --total-timesteps 400000000 --checkpoint-root checkpoints/scoreg
python -m scripts.torch_scoreg_layout.train_struct_noloc --social-network fc --comm_field 100 --no-agent-visible --num_networks 3 --seed 2 --no-self-play-option --total-timesteps 400000000 --checkpoint-root checkpoints/scoreg
python -m scripts.torch_scoreg_layout.train_struct_noloc --social-network fc --comm_field 100 --no-agent-visible --num_networks 3 --seed 3 --no-self-play-option --total-timesteps 400000000 --checkpoint-root checkpoints/scoreg
