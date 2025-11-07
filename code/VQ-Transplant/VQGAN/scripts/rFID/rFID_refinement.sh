#!/bin/bash
#SBATCH --job-name=rFID_wasserstein_refinement
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g015
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant3/metrics/Refinement/ImageNet/rFID_wasserstein_refinement.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant3/metrics/Refinement/ImageNet/rFID_wasserstein_refinement.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/FID
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_16384_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_16384_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_16384_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_16384_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_16384_False_5.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_32768_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_32768_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_32768_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_32768_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_32768_False_5.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_65536_False_1.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_65536_False_2.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_65536_False_3.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_65536_False_4.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant3/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/ImageNet --sample_name wasserstein_vq_refinement_65536_False_5.npz