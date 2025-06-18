#!/bin/bash
#SBATCH --job-name=eval_vqgan_p3
#SBATCH --partition=gpuv100
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQGAN2/reconstruction/FFHQ/eval_vqgan_result_p3.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQGAN2/reconstruction/FFHQ/eval_vqgan_error_p3.out

conda activate share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=25667 eval_reconstruction.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 100000  --beta=0.1 --gamma=0.5