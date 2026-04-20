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

# python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 8 --seed 1 --total-timesteps 1000000000
# python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 3 --seed 1 --total-timesteps 600000000
python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 2 --seed 1 --total-timesteps 300000000

python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 8 --seed 2 --total-timesteps 500000000
python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 3 --seed 2 --total-timesteps 300000000
python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 2 --seed 2 --total-timesteps 300000000

python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 8 --seed 3 --total-timesteps 500000000
python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 3 --seed 3 --total-timesteps 300000000
python -m scripts.torch_scoreg_layout.train --comm_field 100 --image_size 3 --no-agent-visible --num_networks 2 --seed 3 --total-timesteps 300000000