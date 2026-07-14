#!/bin/bash
#SBATCH --job-name=rfid_vqgan_refinement_p2
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b2
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/metrics/Refinement/ImageNet/rfid_vqgan_refinement_p2.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/metrics/Refinement/ImageNet/rfid_vqgan_refinement_p2.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_6.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_7.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_8.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_9.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_10_10.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_6.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_7.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_8.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_9.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_10.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_11.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_12.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_13.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_14.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/ImageNet/ --sample_name wasserstein_vq_refinement_32768_False_15_15.npz