#!/bin/bash
#SBATCH --job-name=online_quantizer
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=45G
#SBATCH -c 6
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --output /h/.../analysis/Uniform/online_result.out
#SBATCH --error /h/.../analysis/Uniform/online_error.out

conda activate share_VAR 
python VariousVQs.py --vector_quantizer=online_clustering --mean 0.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 1.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 2.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 3.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 4.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 5.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 6.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 7.0
python VariousVQs.py --vector_quantizer=online_clustering --mean 8.0