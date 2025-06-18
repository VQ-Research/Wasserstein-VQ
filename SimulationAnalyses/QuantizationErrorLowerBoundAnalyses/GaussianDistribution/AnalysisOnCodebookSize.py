import numpy as np
import math
from scipy.spatial.distance import cdist
from FID import calculate_frechet_distance
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

codebook_size_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
feature_size_list = [10000, 50000, 100000, 200000, 300000, 500000, 800000, 1000000]
embed_dim_list = [2, 4, 8, 16, 32, 64, 128, 256]
mean_list = [2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
variance_list = [36.0, 25.0, 16.0, 9.0, 4.0, 1.0]
embed_dim = 32
feature_size = 200000


repeat_times = 5
total_wasserstein_distance_list = []
total_codebook_utilization_list = []
total_perplexity_list = []
total_distance_list = []
for k in range(repeat_times):
    wasserstein_distance_list = [[] for i in range(len(codebook_size_list))]
    codebook_utilization_list = [[] for i in range(len(codebook_size_list))]
    perplexity_list = [[] for i in range(len(codebook_size_list))]
    distance_list = [[] for i in range(len(codebook_size_list))]

    for i in range(len(codebook_size_list)):
        codebook_size = codebook_size_list[i]

        for mean in mean_list:
            Gaussian = torch.distributions.multivariate_normal.MultivariateNormal(mean * torch.ones(embed_dim,device=device), torch.eye(embed_dim, device=device))
            
            feature = Gaussian.sample([feature_size])
            feature_mean = feature.mean(0)
            feature_covariance = torch.mm((feature - torch.mean(feature, dim=0, keepdim=True)).t(), feature - torch.mean(feature, dim=0, keepdim=True))/feature_size            
            
            codebook = Gaussian.sample([codebook_size])
            codebook_mean = codebook.mean(0)
            codebook_covariance = torch.mm((codebook - torch.mean(codebook, dim=0, keepdim=True)).t(), codebook - torch.mean(codebook, dim=0, keepdim=True))/codebook_size

            wasserstein_distance = calculate_frechet_distance(feature_mean.cpu().numpy(), feature_covariance.cpu().numpy(), codebook_mean.cpu().numpy(), codebook_covariance.cpu().numpy())
            wasserstein_distance_list[i].append(wasserstein_distance)

            dist = torch.sum(feature.square(), dim=1, keepdim=True) + torch.sum(codebook.square(), dim=1, keepdim=False)
            dist.addmm_(feature, codebook.T, alpha=-2, beta=1)
            idx = torch.argmin(dist, dim=1)

            nearest_distances = dist.gather(1, idx.unsqueeze(1)).squeeze(1)
            avg_nearest_distance = nearest_distances.mean().item()
            distance_list[i].append(avg_nearest_distance)

            codebook_histogram = idx.bincount(minlength=codebook_size).float()
            codebook_usage_counts = (codebook_histogram > 0).float().sum()

            codebook_utilization = codebook_usage_counts.item() / codebook_size
            codebook_utilization_list[i].append(codebook_utilization)
            avg_probs = codebook_histogram/feature_size

            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
            perplexity_list[i].append(perplexity)

            print("codebook size: "+str(codebook_size)+"  mean: "+str(mean)+"  wasserstein_distance: "+str(wasserstein_distance)+"  codebook_utilization: "+str(codebook_utilization)+"   perplexity: "+str(perplexity))

    total_wasserstein_distance_list.append(wasserstein_distance_list)
    total_codebook_utilization_list.append(codebook_utilization_list)
    total_perplexity_list.append(perplexity_list)
    total_distance_list.append(distance_list)

total_wasserstein_distance_list_mean = total_wasserstein_distance_list
total_codebook_utilization_list_mean = total_codebook_utilization_list
total_perplexity_list_mean = total_perplexity_list
total_disntance_list_mean = total_distance_list



total_wasserstein_distance_list = []
total_codebook_utilization_list = []
total_perplexity_list = []
total_distance_list = []
for k in range(repeat_times):
    wasserstein_distance_list = [[] for i in range(len(codebook_size_list))]
    codebook_utilization_list = [[] for i in range(len(codebook_size_list))]
    perplexity_list = [[] for i in range(len(codebook_size_list))]
    distance_list = [[] for i in range(len(codebook_size_list))]

    for i in range(len(codebook_size_list)):
        codebook_size = codebook_size_list[i]

        for variance in variance_list:
            Gaussian = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(embed_dim), variance * torch.eye(embed_dim))
            
            feature = Gaussian.sample([feature_size])
            feature_mean = feature.mean(0)
            feature_covariance = torch.mm((feature - torch.mean(feature, dim=0, keepdim=True)).t(), feature - torch.mean(feature, dim=0, keepdim=True))/feature_size
            
            codebook = Gaussian.sample([codebook_size])
            codebook_mean = codebook.mean(0)
            codebook_covariance = torch.mm((codebook - torch.mean(codebook, dim=0, keepdim=True)).t(), codebook - torch.mean(codebook, dim=0, keepdim=True))/codebook_size

            wasserstein_distance = calculate_frechet_distance(feature_mean.numpy(), feature_covariance.numpy(), codebook_mean.numpy(), codebook_covariance.numpy())
            wasserstein_distance_list[i].append(wasserstein_distance)


            dist = torch.sum(feature.square(), dim=1, keepdim=True) + torch.sum(codebook.square(), dim=1, keepdim=False)
            dist.addmm_(feature, codebook.T, alpha=-2, beta=1)
            idx = torch.argmin(dist, dim=1)

            nearest_distances = dist.gather(1, idx.unsqueeze(1)).squeeze(1)
            avg_nearest_distance = nearest_distances.mean().item()
            distance_list[i].append(avg_nearest_distance)

            codebook_histogram = idx.bincount(minlength=codebook_size).float()
            codebook_usage_counts = (codebook_histogram > 0).float().sum()

            codebook_utilization = codebook_usage_counts.item() / codebook_size
            codebook_utilization_list[i].append(codebook_utilization)

            avg_probs = codebook_histogram/feature_size

            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
            perplexity_list[i].append(perplexity)

            print("codebook size: "+str(codebook_size)+"  variance: "+str(variance)+"  wasserstein_distance: "+str(wasserstein_distance)+"  codebook_utilization: "+str(codebook_utilization)+"   perplexity: "+str(perplexity))
        
    total_wasserstein_distance_list.append(wasserstein_distance_list)
    total_codebook_utilization_list.append(codebook_utilization_list)
    total_perplexity_list.append(perplexity_list)
    total_distance_list.append(distance_list)
    
with open('Data_pkl/lower_bound_Gaussian_CodebookSize.pkl', 'wb') as f:
    pickle.dump([total_wasserstein_distance_list_mean, total_codebook_utilization_list_mean, total_perplexity_list_mean, total_disntance_list_mean,
                 total_wasserstein_distance_list, total_codebook_utilization_list, total_perplexity_list, total_distance_list], f)