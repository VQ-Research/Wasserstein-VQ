#!/bin/bash
#SBATCH --job-name=p1_wasserstein_imagenet
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQVAE/slurm/ImageNet/wasserstein_result_p1.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQVAE/slurm/ImageNet/wasserstein_error_p1.out

conda activate share_VAR 
CUDA_VISIBLE_DEVICES="0" python train_vqvae.py --quantizer_name=wasserstein_quantizer --dataset_name=ImageNet --batch_size 32 --factor 16 --codebook_size 16384  --beta=0.1 --gamma=0.5 --alpha=1.0