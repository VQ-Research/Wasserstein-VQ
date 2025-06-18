#!/bin/bash
#SBATCH --job-name=vqgan_ffhq_p2
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:4
#SBATCH --time=5-00:00:00
#SBATCH --output /mmfs1/data/fangxian/WassersteinVQ/VQGAN2/slurm/FFHQ/vqgan_ffhq_result_p2.out
#SBATCH --error /mmfs1/data/fangxian/WassersteinVQ/VQGAN2/slurm/FFHQ/vqgan_ffhq_error_p2.out

conda activate share_VAR 
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 train_vqgan.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 16384  --beta=0.1 --gamma=0.5
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 train_vqgan.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 50000  --beta=0.1 --gamma=0.5
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 train_vqgan.py --quantizer_name=wasserstein_quantizer --dataset_name=FFHQ --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 100000 --beta=0.1 --gamma=0.5