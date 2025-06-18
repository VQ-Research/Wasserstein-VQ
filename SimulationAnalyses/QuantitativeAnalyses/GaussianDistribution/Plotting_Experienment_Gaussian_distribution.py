import numpy as np
import math
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F
from scipy.io import savemat
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

device = 'cpu'

codebook_size_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
feature_size_list = [10000, 50000, 100000, 200000, 300000, 500000, 800000, 1000000]
embed_dim_list = [2, 4, 8, 16, 32, 64, 128, 256]
mean_list = [2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
variance_list = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

###########################  Plotting  ###########################
def Calculate_and_Draw(total_list = None, Y_Name = None, X_Name = None, LOG = False, save_path = None):
    total_list = np.array(total_list)
    if LOG == True:
        total_list = np.log10(total_list)
    
    mu_list = np.mean(total_list, axis=0)
    std_list = np.std(total_list, axis=0, ddof=1)
    confidence_interval_lower = mu_list - 1.96 * std_list
    confidence_interval_upper = mu_list + 1.96 * std_list
    
    if Y_Name == 'Codebook Utilization':
        for i in range(len(confidence_interval_upper)):
            for j in range(len(confidence_interval_upper[0])):
                if confidence_interval_upper[i,j] > 1:
                    confidence_interval_upper[i,j] = 1

    colors = [(0, 63, 92), (47, 75, 124), (102, 81, 145), (160, 81, 149), (212, 80, 135), (249, 93, 106), (255, 124, 67), (255, 166, 0)]
    colors = [(r/255, g/255, b/255) for r, g, b in colors]
    
    plt.figure(figsize=(6.4, 5.0))
    positions = [
        (135.0/1037.0, 126.0/808.0, 874.0/1037.0, 635.0/808.0), 
    ]
    plt.axes(positions[0]) 
    
    for i in range(len(mu_list)):
        
        if X_Name != 'Mean' and Y_Name == 'Quantization Error' and 'CodebookSize' in save_path :
            print(10 ** np.array(mu_list[i][-1]))
        
        if X_Name == 'Mean':
            x = np.array(mean_list)
        else :
            x = np.array(variance_list)
            
        if 'CodebookSize' in save_path:
            plt.plot(x, mu_list[i], marker='o', markersize=6, linestyle='-', label=f'K = {codebook_size_list[i]}', color=colors[i])
        elif 'FeatureSize' in save_path:
            plt.plot(x, mu_list[i], marker='o', markersize=6, linestyle='-', label=f'N = {feature_size_list[i]}', color=colors[i])
        elif 'FeatureDim' in save_path:
            plt.plot(x, mu_list[i], marker='o', markersize=6, linestyle='-', label=f'd = {embed_dim_list[i]}', color=colors[i])
        else:
            raise ValueError("Invaild save_path")
        
        plt.fill_between(x, confidence_interval_lower[i], confidence_interval_upper[i], alpha=0.2, color=colors[i])
    
    x_range = 0
    if X_Name == 'Mean':
        plt.xlabel(r'$\mu$', fontsize=22)
        plt.xlim(-0.03, 2.53)
        plt.xticks(np.linspace(0, 2.5, 6))
        x_range = 2.5
    else :
        plt.xlabel(r'$\sigma$', fontsize=22)
        plt.xlim(0.96, 6.06)
        plt.xticks(np.linspace(1, 6, 6))
        x_range = 5
    
    y_range = 0
    if Y_Name == 'Codebook Utilization':
        plt.ylabel(r'Codebook Utilization ($\mathcal{U}$)', fontsize=22)
        plt.ylim(0.0, 1.02)
        plt.yticks(np.linspace(0.0, 1.0, 6))
        y_range = 1.02
    elif Y_Name == 'PPL (log)':
        plt.ylabel(r'Codebook Perplexity ($\log$$\mathcal{C}$)', fontsize=22)
        plt.ylim(-1, 4.05)
        plt.yticks(np.linspace(-1, 4.0, 6))
        y_range = 5.05
    elif Y_Name == 'Quantization Error':
        plt.ylabel(r'Quantization Error ($\log$$\mathcal{E}$)', fontsize=22)
        if 'FeatureDim' in save_path and 'lower_bound' not in save_path:
            plt.ylim(-5, 5.1)
            plt.yticks(np.linspace(-5, 5.0, 6))
            y_range = 10.1
        elif 'FeatureDim' in save_path and 'lower_bound' in save_path:
            plt.ylim(-5, 5.10)
            plt.yticks(np.linspace(-5, 5.0, 6))
            y_range = 5.10 - (-5)
        elif 'FeatureDim' not in save_path and 'lower_bound' in save_path:
            if X_Name == 'Mean':
                plt.ylim(0.0, 2.02)
                plt.yticks(np.linspace(0.0, 2.0, 6))  
                y_range = 2.02
            else :
                plt.ylim(0.0, 4.04)
                plt.yticks(np.linspace(0.0, 4, 6))
                y_range = 4.04
        else :
            plt.ylim(0.5, 3.03)
            plt.yticks(np.linspace(0.5, 3.0, 6))
            y_range = 3.03-0.5
    else :
        plt.ylabel(Y_Name, fontsize=22)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if Y_Name == 'Quantization Error':
        plt.legend(loc='lower right')
        plt.grid(True)
    else :
        plt.legend(loc='upper right')
        plt.grid(True)        

    plt.gca().spines['right'].set_visible(1.5)
    plt.gca().spines['top'].set_visible(1.5)
    plt.gca().spines['left'].set_linewidth(0.67)
    plt.gca().spines['bottom'].set_linewidth(0.67)

    aspect_ratio = 0.75 * x_range / y_range
    plt.gca().set_aspect(aspect_ratio)

    plt.tight_layout()
    plt.savefig(f'Experiment_Data/FIG_4/{save_path}.pdf')
    plt.clf()
    
    if LOG == True:
        mu_list = np.power(10, mu_list)
    return mu_list

###########################  Loading Experienment on Gaussian Distribution  ###########################
Plotting_CodebookSize = True  
Plotting_FeatureDim = True
Plotting_FeatureSize = True

if Plotting_CodebookSize:
    with open('Data_pkl/Gaussian_CodebookSize.pkl', 'rb') as f:
        lists = pickle.load(f)
    list1, list2, list3, list4, list5, list6, list7, list8 = lists
    
    utili_list_mean = Calculate_and_Draw(total_list = list2, Y_Name = 'Codebook Utilization', X_Name = 'Mean', save_path='CodebookSize_mean_Utili')
    ppl_list_mean = Calculate_and_Draw(total_list = list3, Y_Name = 'PPL (log)', X_Name = 'Mean', LOG=True, save_path='CodebookSize_mean_ppl')
    disntance_list = Calculate_and_Draw(total_list = list4, Y_Name = 'Quantization Error', X_Name = 'Mean', LOG=True, save_path='CodebookSize_mean_Quant')

    utili_list = Calculate_and_Draw(total_list = list6, Y_Name = 'Codebook Utilization', X_Name = 'Variance', save_path='CodebookSize_Var_Util')
    ppl_list = Calculate_and_Draw(total_list = list7, Y_Name = 'PPL (log)', X_Name = 'Variance', LOG=True, save_path='CodebookSize_Var_ppl')
    disntance_list = Calculate_and_Draw(total_list = list8, Y_Name = 'Quantization Error', X_Name = 'Variance', LOG=True, save_path='CodebookSize_Var_Quant')

if Plotting_FeatureDim:
    name = 'FeatureDim'
    with open('Data_pkl/Gaussian_FeatureDim.pkl', 'rb') as f:
        lists = pickle.load(f)
    list1, list2, list3, list4, list5, list6, list7, list8 = lists

    utili_list_mean = Calculate_and_Draw(total_list = list2, Y_Name = 'Codebook Utilization', X_Name = 'Mean', save_path=f'{name}_mean_Utili')
    ppl_list_mean = Calculate_and_Draw(total_list = list3, Y_Name = 'PPL (log)', X_Name = 'Mean', LOG=True, save_path=f'{name}_mean_ppl')
    disntance_list = Calculate_and_Draw(total_list = list4, Y_Name = 'Quantization Error', X_Name = 'Mean', LOG=True, save_path=f'{name}_mean_Quant')

    utili_list = Calculate_and_Draw(total_list = list6, Y_Name = 'Codebook Utilization', X_Name = 'Variance', save_path=f'{name}_Var_Util')
    ppl_list = Calculate_and_Draw(total_list = list7, Y_Name = 'PPL (log)', X_Name = 'Variance', LOG=True, save_path=f'{name}_Var_ppl')
    disntance_list = Calculate_and_Draw(total_list = list8, Y_Name = 'Quantization Error', X_Name = 'Variance', LOG=True, save_path=f'{name}_Var_Quant')
     
if Plotting_FeatureSize:
    name = 'FeatureSize'
    with open('Data_pkl/Gaussian_FeatureSize.pkl', 'rb') as f:
        lists = pickle.load(f)
    list1, list2, list3, list4, list5, list6, list7, list8 = lists

    utili_list_mean = Calculate_and_Draw(total_list = list2, Y_Name = 'Codebook Utilization', X_Name = 'Mean', save_path=f'{name}_mean_Utili')
    ppl_list_mean = Calculate_and_Draw(total_list = list3, Y_Name = 'PPL (log)', X_Name = 'Mean', LOG=True, save_path=f'{name}_mean_ppl')
    disntance_list = Calculate_and_Draw(total_list = list4, Y_Name = 'Quantization Error', X_Name = 'Mean', LOG=True, save_path=f'{name}_mean_Quant')

    utili_list = Calculate_and_Draw(total_list = list6, Y_Name = 'Codebook Utilization', X_Name = 'Variance', save_path=f'{name}_Var_Util')
    ppl_list = Calculate_and_Draw(total_list = list7, Y_Name = 'PPL (log)', X_Name = 'Variance', LOG=True, save_path=f'{name}_Var_ppl')
    disntance_list = Calculate_and_Draw(total_list = list8, Y_Name = 'Quantization Error', X_Name = 'Variance', LOG=True, save_path=f'{name}_Var_Quant')
