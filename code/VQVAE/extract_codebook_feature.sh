#!/bin/bash
#SBATCH --job-name=extract_data
#SBATCH --partition=gpuv100
#SBATCH --nodes=1
#SBATCH --mem=30gb
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQVAE/extract_data/extract_data_result.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQVAE/extract_data/extract_data_error.out

CUDA_VISIBLE_DEVICES="0" python extract_codebook_feature.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --batch_size 32 --factor 16 --codebook_size 8192 --beta=1.0 --gamma=0.5
CUDA_VISIBLE_DEVICES="0" python extract_codebook_feature.py --quantizer_name=vanilla_quantizer --dataset_name=FFHQ --batch_size 32 --factor 16 --codebook_size 8192 --beta=1.0 --gamma=0.0 
CUDA_VISIBLE_DEVICES="0" python extract_codebook_feature.py --quantizer_name=online_quantizer --dataset_name=FFHQ --batch_size 32 --factor 16 --codebook_size 8192 --beta=1.0 --gamma=0.0 
CUDA_VISIBLE_DEVICES="0" python extract_codebook_feature.py --quantizer_name=ema_quantizer --dataset_name=FFHQ --batch_size 32 --factor 16 --codebook_size 8192 --beta=1.0 --gamma=0.0 
