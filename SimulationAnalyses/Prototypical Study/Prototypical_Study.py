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

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'Times New Roman'

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
set_seed(7)

device = 'cpu'

mean_list = [-0.6, -0.7, 0.0, 0.7, 1.4]
r_list = [0.4, 0.8, 1.0, 1.1, 1.3, 1.6]
feature_size = 10000
codebook_size = 400

def GenCircleRandomPts(r, point_cnt, center_x, center_y):        
    x_list = []
    y_list = []
    for k in range(point_cnt):
        random_theta = random.random() * math.pi * 2
        random_r = math.sqrt(random.random()) * r
        x = math.cos(random_theta) * random_r + center_x
        y = math.sin(random_theta) * random_r + center_y
        x_list.append(x)
        y_list.append(y)
    return [x_list, y_list]



def Draw_point(ax, point_list=None, center_x=None, center_y=None, radius=None, label='Points', color_r=None, color_p=None, droput=0.0, point_size=4, pltwidth=2.5):
    circle = plt.Circle((center_x, center_y), radius, fill=True, alpha=0.2, label='Circle', color=color_r)
    ax.add_patch(circle)    
    
    if point_list is not None and droput > 0:
        mask = np.random.rand(len(point_list[0])) > droput
        point_list = [np.array(point_list[0])[mask], np.array(point_list[1])[mask]]
    
    ax.scatter(point_list[0], point_list[1], label=label, color=color_p, s=point_size)

    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')


def Draw_point_R(ax, point_list=None, center_x=None, center_y=None, radius=None, label='Points', color_r=None, color_p=None, droput=0.0, point_size=4):
    circle = plt.Circle((center_x, center_y), radius, fill=True, alpha=0.2, label='Circle', color=color_r)
    ax.add_patch(circle)    
    
    if point_list is not None and droput > 0:
        mask = np.random.rand(len(point_list[0])) > droput
        point_list = [np.array(point_list[0])[mask], np.array(point_list[1])[mask]]
    
    ax.scatter(point_list[0], point_list[1], label=label, color=color_p, s=point_size)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')

color_r1 = '#D20103' # red
color_p1 = '#DA4444'
color_r2 = '#57BA55' # green
color_p2 = '#0C5F16'

repeat_times = 5
width = 1
Feature_Point_list = GenCircleRandomPts(r=1.0, point_cnt=feature_size, center_x=1.4, center_y=0.0)
Mean_Dist_list = []
codebook_utilization_list = []
perplexity_list = []

Feature_Point_array = np.array(Feature_Point_list)

i= 0
for mean in mean_list:
    
    Codebook = GenCircleRandomPts(r=1.0, point_cnt=codebook_size, center_x=mean, center_y=0.0)
    Codebook_array = np.array(Codebook)
    Feature_Point_index = []
    Feature_Point_Nearest_Dist = []
    
    for i in range(Feature_Point_array.shape[1]):
        distances = np.sum((Feature_Point_array[:, i].reshape(-1, 1) - Codebook_array) ** 2, axis=0)
        Codebook_index = np.argmin(distances)
        Min_dist = distances[Codebook_index]
        Feature_Point_index.append(Codebook_index)
        Feature_Point_Nearest_Dist.append(Min_dist)
    mean_dist = sum(Feature_Point_Nearest_Dist) / len(Feature_Point_Nearest_Dist)
    Mean_Dist_list.append(np.sqrt(mean_dist))
    
    idx = torch.tensor(Feature_Point_index)
    codebook_histogram = idx.bincount(minlength=codebook_size).float()
    
    codebook_usage_counts = (codebook_histogram > 0).float().sum()
    codebook_utilization = codebook_usage_counts.item() / codebook_size
    codebook_utilization_list.append(codebook_utilization)
    
    feature_size_temp = len(Feature_Point_list[0])
    avg_probs = codebook_histogram / feature_size_temp
    
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
    perplexity_list.append(perplexity)

    plt.clf()
    fig, ax = plt.subplots(figsize=(15, 7.5))
    Draw_point(ax, point_list=Feature_Point_list, center_x=1.4,
               center_y=0.0, radius=1.0, label='Feature Points',
               color_r=color_r1, color_p=color_p1, droput=0.9, point_size=15)
    Draw_point(ax, point_list=Codebook, center_x=mean, center_y=0.0,
               radius=1.0, label='Codebook Points',
               color_r=color_r2, color_p=color_p2, droput=0.1, point_size=15)

    pltwidth = 2.0
    x_mid = (1.4+mean)/2
    ax.set_xlim(x_mid - pltwidth , pltwidth + x_mid)
    ax.set_ylim(-1.0, 1.0)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig(f"Quantative_Simul_Mean={mean}.pdf", bbox_inches='tight', pad_inches=0.02)

for i in range(len(Mean_Dist_list)):
    print(f'r : {mean_list[i]}')
    print(f"Mean_Dist_list {Mean_Dist_list[i]}")
    print(f"codebook_utilization_list {codebook_utilization_list[i]}")
    print(f"perplexity_list {perplexity_list[i]}")


mean = 0
Feature_Point_list = GenCircleRandomPts(r=1.0, point_cnt=feature_size, center_x=0.0, center_y=0.0)
Feature_Point_array = np.array(Feature_Point_list)

Mean_Dist_list = []
codebook_utilization_list = []
perplexity_list = []

for r in r_list:

    Codebook = GenCircleRandomPts(r=r, point_cnt=codebook_size, center_x=mean, center_y=0.0)
    Codebook_array = np.array(Codebook)
    
    Feature_Point_index = []
    Feature_Point_Nearest_Dist = []
    for i in range(Feature_Point_array.shape[1]):
        distances = np.sum((Feature_Point_array[:, i].reshape(-1, 1) - Codebook_array) ** 2, axis=0)
        Codebook_index = np.argmin(distances)
        Min_dist = distances[Codebook_index]
        Feature_Point_index.append(Codebook_index)
        Feature_Point_Nearest_Dist.append(Min_dist)
    mean_dist = sum(Feature_Point_Nearest_Dist) / len(Feature_Point_Nearest_Dist)
    Mean_Dist_list.append(np.sqrt(mean_dist))
    
    idx = torch.tensor(Feature_Point_index)
    codebook_histogram = idx.bincount(minlength=codebook_size).float()
    
    codebook_usage_counts = (codebook_histogram > 0).float().sum()
    codebook_utilization = codebook_usage_counts.item() / codebook_size
    codebook_utilization_list.append(codebook_utilization)
    
    feature_size_temp = len(Feature_Point_list[0])
    avg_probs = codebook_histogram / feature_size_temp
    
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
    perplexity_list.append(perplexity)

    plt.clf()
    fig, ax = plt.subplots(figsize=(15, 12))
    Draw_point_R(ax, point_list=Feature_Point_list, center_x=0.0, center_y=0.0, radius=1.0, label='Feature Points', color_r=color_r1, color_p=color_p1, droput=0.9, point_size=15)
    Draw_point_R(ax, point_list=Codebook, center_x=mean, center_y=0.0, radius=r, label='Codebook Points', color_r=color_r2, color_p=color_p2, droput=0.1, point_size=15)


    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.6, 1.6)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"Quantative_Simul_R={r}.pdf", bbox_inches='tight', pad_inches=0.03)


for i in range(len(Mean_Dist_list)):
    print(f'r : {r_list[i]}')
    print(f"Mean_Dist_list {Mean_Dist_list[i]}")
    print(f"codebook_utilization_list {codebook_utilization_list[i]}")
    print(f"perplexity_list {perplexity_list[i]}")
