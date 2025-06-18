#!/bin/bash
#SBATCH --job-name=var_eval_p1
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[007-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output /projects/yuanai/projects/WassersteinVQ/VAR/reconstruction/FFHQ/var_eval_result_p1.out
#SBATCH --error /projects/yuanai/projects/WassersteinVQ/VAR/reconstruction/FFHQ/var_eval_error_p1.out

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=35531 eval_reconstruction.py --dataset_name=FFHQ --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 16384  --beta=0.1 --gamma=0.5