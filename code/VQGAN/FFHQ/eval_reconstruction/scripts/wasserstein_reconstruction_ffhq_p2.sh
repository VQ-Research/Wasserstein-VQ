#!/bin/bash
#SBATCH --job-name=eval_vqgan_p2
#SBATCH --partition=gpuv100
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQGAN2/reconstruction/FFHQ/eval_vqgan_result_p2.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQGAN2/reconstruction/FFHQ/eval_vqgan_error_p2.out

conda activate share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=25666 eval_reconstruction.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 50000  --beta=0.1 --gamma=0.5