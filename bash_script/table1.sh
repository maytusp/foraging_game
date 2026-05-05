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




python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 2 --model-name pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 2 --model-name pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 2 --model-name pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 3

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 2 --model-name sp_pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 2 --model-name sp_pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 2 --model-name sp_pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 3


python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 3 --model-name pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 3 --model-name pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 3 --model-name pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 3 

python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 3 --model-name sp_pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 3 --model-name sp_pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.prepare_ls_traj --all-pairs --num-networks 3 --model-name sp_pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/langsim/ --total-episodes 1000 --seed 3




python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 2 --model-name pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 2 --model-name pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 2 --model-name pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 3

python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 2 --model-name sp_pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 2 --model-name sp_pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 2 --model-name sp_pop_ppo_2net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 3


python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 3 --model-name pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 3 --model-name pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 3 --model-name pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 3

python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 3 --model-name sp_pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 1
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 3 --model-name sp_pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 2
python -m scripts.torch_scoreg_layout.eval_batched_sr --all-pairs --num-networks 3 --model-name sp_pop_ppo_3net_invisible --model-step 99993600 --ckpt-root checkpoints/torch_scoreg_layout2 --save-root logs/scoreg/sr/ --total-episodes 1000 --seed 3


