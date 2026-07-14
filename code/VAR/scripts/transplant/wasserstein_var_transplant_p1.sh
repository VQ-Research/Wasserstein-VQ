#!/bin/bash
#SBATCH --job-name=wasserstein_var_transplant_p1
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b3,gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/slurm/Transplant/ImageNet/wasserstein_var_transplant_p1.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/slurm/Transplant/ImageNet/wasserstein_var_transplant_p1.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12261 train_VAR_transplant.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size=4096  --codebook_dim=32 --use_multiscale --stage=transplant --transplant_epochs=5 --alpha=1.0 --beta=1.0 --gamma=0.2