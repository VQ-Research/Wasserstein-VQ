#!/bin/bash
#SBATCH --job-name=p3_wasserstein_imagenet
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --mem=40gb
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQVAE/slurm/ImageNet/wasserstein_result_p3.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQVAE/slurm/ImageNet/wasserstein_error_p3.out

conda activate share_VAR 
CUDA_VISIBLE_DEVICES="0" python train_vqvae.py --quantizer_name=wasserstein_quantizer --dataset_name=ImageNet --batch_size 32 --factor 16 --codebook_size 100000 --beta=0.1 --gamma=0.5 --alpha=1.0