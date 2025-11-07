#!/bin/bash
#SBATCH --job-name=rfid_var_refinement_p1
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b2
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/metrics/Refinement/ImageNet/rfid_var_refinement_p1.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/metrics/Refinement/ImageNet/rfid_var_refinement_p1.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_6.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_7.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_8.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_9.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_10_10.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_6.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_7.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_8.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_9.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_10.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_11.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_12.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_13.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_14.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VAR/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_4096_True_15_15.npz
