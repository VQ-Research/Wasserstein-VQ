#!/bin/bash
#SBATCH --job-name=linear_quantizer
#SBATCH --partition=a40
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=45G
#SBATCH -c 6
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --output /h/.../analysis/Uniform/linear_result.out
#SBATCH --error /h/.../analysis/Uniform/linear_error.out

conda activate share_VAR 
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 0.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 1.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 2.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 3.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 4.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 5.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 6.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 7.0
python VariousVQs.py --vector_quantizer=linear_quantizer --mean 8.0