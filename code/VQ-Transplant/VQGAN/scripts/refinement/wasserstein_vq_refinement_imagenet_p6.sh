#!/bin/bash
#SBATCH --job-name=wasserstein_vq_refinement_imagenet_p6
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/slurm/Refinement/ImageNet/wasserstein_vq_refinement_imagenet_p6.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/slurm/Refinement/ImageNet/wasserstein_vq_refinement_imagenet_p6.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12856 train_VQ_refinement.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size=65536  --codebook_dim=16 --stage=refinement --refinement_epochs=15 --alpha=1.0 --beta=0.2 --gamma=1.0 --disc_weight=0.4 --checkpoint_name checkpoint-wasserstein_vq_transplant_False_ImageNet_model_65536_16_loss_1.0_0.2_1.0_0.4-5.pth.tar
