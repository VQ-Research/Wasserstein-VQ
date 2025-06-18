#!/bin/bash
#SBATCH --job-name=p2_wasserstein_ffhq
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --mem=10gb
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQVAE/slurm/FFHQ/wasserstein_result_p2.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQVAE/slurm/FFHQ/wasserstein_error_p2.out

conda activate share_VAR 
CUDA_VISIBLE_DEVICES="0" python train_vqvae.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --batch_size 32 --factor 16 --codebook_size 2048   --beta=0.1 --gamma=0.5 --alpha=1.0