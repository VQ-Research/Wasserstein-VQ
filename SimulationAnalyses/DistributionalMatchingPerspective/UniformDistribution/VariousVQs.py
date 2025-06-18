import os
import sys
import time
import torch
import argparse
import numpy as np
import math
import torch.nn as nn
from torch import nn, optim
import random
import scipy.io as sio
from torch.nn import functional as F
from scipy.io import savemat
import torchvision
import pandas as pd

def parse_arg():
    parser = argparse.ArgumentParser(description='Various Vector Quantizers (Uniform Distribution).') 
    parser.add_argument('--seed', type=int, default=12, metavar='S', help='random seed.')
    parser.add_argument('--vector_quantizer', default='wasserstein_quantizer', help='the types of vector quantizer.', choices=['wasserstein_quantizer', 'vanilla_quantizer', 'ema_quantizer', 'linear_quantizer', 'online_clustering'])
    parser.add_argument('--max_steps', type=int, default=2000, help="training steps.")
    parser.add_argument('--codebook_size', type=int, default=16384, help="codebook size.")
    parser.add_argument('--codebook_dim', type=int, default=8, help="codebook dim.")
    parser.add_argument('--train_samples', type=int, default=50000, help="train samples.")
    parser.add_argument('--val_samples', type=int, default=200000, help="val steps.")
    parser.add_argument("--lr", type=float, default=3.0, help="learning rate")
    parser.add_argument("--mean", type=float, default=2.0, help="the deviation to the codebook distribution")
    args = parser.parse_args()
    if args.vector_quantizer == "wasserstein_quantizer":
        args.lr = 3.0
    elif args.vector_quantizer == "vanilla_quantizer":
        args.lr = 1.0
    elif args.vector_quantizer == "linear_quantizer":
        args.lr = 0.02
    elif args.vector_quantizer == "online_clustering":
        args.lr = 3.0

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return args

class VectorQuantizer(nn.Module):
    def __init__(self, args):
        super(VectorQuantizer, self).__init__()
        initial = torch.randn(args.codebook_size, args.codebook_dim).uniform_(-1.0, 1.0)
        self.embedding = nn.Embedding(args.codebook_size, args.codebook_dim)
        self.embedding.weight.data.copy_(initial)
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        if args.vector_quantizer == "wasserstein_quantizer" or args.vector_quantizer == "vanilla_quantizer" or args.vector_quantizer == "online_clustering":
            self.embedding.weight.requires_grad = True
        else:
            self.embedding.weight.requires_grad = False

        if args.vector_quantizer == "linear_quantizer":
            self.codebook_projection = torch.nn.Sequential(torch.nn.Linear(args.codebook_dim, args.codebook_dim))

        if args.vector_quantizer == "ema_quantizer":
            self.eps = 1e-5
            self.decay = 0.9
            self.cluster_size = torch.nn.Parameter(torch.zeros(self.codebook_size), requires_grad = False)
            self.embed_avg = torch.nn.Parameter(self.embedding.weight.clone(), requires_grad = False)
        if args.vector_quantizer == "online_clustering":
            self.decay = 0.9
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    ## produce a loss for wasserstein_quantizer; as the metric for all quantizer
    def calc_wasserstein_distance(self, z):
        if args.vector_quantizer == "linear_quantizer":
            codebook = self.codebook_projection(self.embedding.weight)
        else:
            codebook = self.embedding.weight

        N = z.size(0)
        D = z.size(1)
        codebook_size = self.codebook_size

        z_mean = z.mean(0)
        z_covariance = torch.mm((z - torch.mean(z, dim=0, keepdim=True)).t(), z - torch.mean(z, dim=0, keepdim=True))/N
        
        ### compute the mean and covariance of codebook vectors
        c = codebook
        c_mean = c.mean(0)
        c_covariance = torch.mm((c - torch.mean(c, dim=0, keepdim=True)).t(), c - torch.mean(c, dim=0, keepdim=True))/codebook_size

        ### calculation of part1
        part_mean =  torch.sum(torch.multiply(z_mean - c_mean, z_mean - c_mean))

        d_covariance = torch.mm(z_covariance, c_covariance)
        ### 1/2 d_covariance
        S, Q = torch.linalg.eigh(d_covariance)
        sqrt_S = torch.sqrt(torch.diag(F.relu(S)) + 1e-8)
        d_sqrt_covariance = torch.mm(torch.mm(Q, sqrt_S), Q.T)

        #############calculation of part2
        part_covariance = F.relu(torch.trace(z_covariance + c_covariance - 2.0 * d_sqrt_covariance))
        wasserstein_loss = torch.sqrt(part_mean + part_covariance + 1e-8)
        return wasserstein_loss

    ## produce a loss for linear_quantizer and vanilla_quantizer
    def calc_commit_loss(self, z):
        if args.vector_quantizer == "linear_quantizer":
            codebook = self.codebook_projection(self.embedding.weight)
        else:
            codebook = self.embedding.weight

        distance = torch.sum(z.detach().square(), dim=1, keepdim=True) + torch.sum(codebook.data.square(), dim=1, keepdim=False)
        distance.addmm_(z.detach(), codebook.data.T, alpha=-2, beta=1)

        token = torch.argmin(distance, dim=1) 
        embed = F.embedding(token, codebook).view(z.shape)
        commit_loss = (embed - z.detach()).square().sum(1).mean()
        return commit_loss

    ## update strategy for the ema_quantizer
    def ema_cluster_size_ema_update(self, new_cluster_size):
        assert args.vector_quantizer == "ema_quantizer"
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
    
    def ema_embed_avg_ema_update(self, new_embed_avg): 
        assert args.vector_quantizer == "ema_quantizer"
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def ema_weight_update(self, num_tokens):
        assert args.vector_quantizer == "ema_quantizer"
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)  

    def ema_update(self, z):
        distance = torch.sum(z.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
        distance.addmm_(z.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

        token = torch.argmin(distance, dim=1)
        embed = self.embedding(token)
        quant_error = (embed - z.detach()).square().sum(1).mean()
        onehot_probs = F.one_hot(token, self.codebook_size).type(z.dtype)

        #EMA cluster size           
        self.ema_cluster_size_ema_update(onehot_probs.sum(0))

        #EMA embedding average
        embed_sum = onehot_probs.transpose(0,1) @ z         
        self.ema_embed_avg_ema_update(embed_sum)

        #normalize embed_avg and update weight
        self.ema_weight_update(self.codebook_size)
        return quant_error

    ## update strategy for the online_clustering
    def online_update(self, z):
        distance = torch.sum(z.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
        distance.addmm_(z.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

        token = torch.argmin(distance, dim=1)
        embed = self.embedding(token)
        quant_error = (embed - z.detach()).square().sum(1).mean()
        onehot_probs = F.one_hot(token, self.codebook_size).type(z.dtype)
        avg_probs = torch.mean(onehot_probs, dim=0)

        self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
        #_, indices = distance.sort(dim=0)
        #random_feat = z.detach()[indices[-1,:]]

        indices = torch.argmin(distance, dim=0)
        ## random selected
        #indices = torch.randint(0, z.size(0), (self.codebook_size,))
        random_feat = z.detach()[indices]
        
        decay = torch.exp(-(self.embed_prob * self.codebook_size * 10)/(1-self.decay) - 1e-3).unsqueeze(1).repeat(1, self.codebook_dim)
        self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
        return quant_error

    ## for all quantizers
    def calc_metrics(self, z):
        if args.vector_quantizer == "linear_quantizer":
            codebook = self.codebook_projection(self.embedding.weight)
        else:
            codebook = self.embedding.weight

        distance = torch.sum(z.detach().square(), dim=1, keepdim=True) + torch.sum(codebook.data.square(), dim=1, keepdim=False)
        distance.addmm_(z.detach(), codebook.data.T, alpha=-2, beta=1)

        token = torch.argmin(distance, dim=1) 
        embed = F.embedding(token, codebook).view(z.shape)

        quant_error = (embed - z.detach()).square().sum(1).mean()
        codebook_histogram = token.bincount(minlength=self.codebook_size).float()
        codebook_usage_counts = (codebook_histogram > 0).float().sum()
        codebook_utilization = codebook_usage_counts.item() / self.codebook_size

        avg_probs = codebook_histogram/codebook_histogram.sum(0)
        codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        wasserstein_distance = self.calc_wasserstein_distance(z)
        return quant_error, codebook_utilization, codebook_perplexity, wasserstein_distance

def main_worker(args):
    quantizer = VectorQuantizer(args).cuda()
    if args.vector_quantizer == "wasserstein_quantizer" or args.vector_quantizer == "vanilla_quantizer" or args.vector_quantizer == "online_clustering":
        optimizer = torch.optim.SGD(quantizer.embedding.parameters(), lr=args.lr, momentum=0.9)       
    elif args.vector_quantizer == "linear_quantizer":
        optimizer = torch.optim.SGD(quantizer.codebook_projection.parameters(), lr=args.lr, momentum=0.9)

    results = {'step':[], 'quant_error':[], 'codebook_utilization':[], 'codebook_perplexity': [], 'wasserstein_distance':[]}
    
    ##### zero-step
    z = torch.randn(args.val_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
    quant_error, codebook_utilization, codebook_perplexity, wasserstein_distance = quantizer.calc_metrics(z)
    results['step'].append(0)
    results['quant_error'].append(quant_error.item())
    results['codebook_utilization'].append(codebook_utilization)
    results['codebook_perplexity'].append(codebook_perplexity.item())
    results['wasserstein_distance'].append(wasserstein_distance.item())

    for step in range(1, args.max_steps+1):
        z = torch.randn(args.train_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
        if args.vector_quantizer == "wasserstein_quantizer":
            commit_loss = quantizer.calc_commit_loss(z)
            wasserstein_loss = quantizer.calc_wasserstein_distance(z)
            loss = 0.1*commit_loss + 10*wasserstein_loss
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            if step == 1 or step%10 == 0:
                print('train step:{}/{}, commit loss:{:.4f}, wasserstein_loss:{:.4f}'.format(step, args.max_steps, commit_loss.item(), wasserstein_loss.item()))
            if step == 1 or step%100 == 0:
                z = torch.randn(args.val_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
                quant_error, codebook_utilization, codebook_perplexity, wasserstein_distance = quantizer.calc_metrics(z)

                results['step'].append(step)
                results['quant_error'].append(quant_error.item())
                results['codebook_utilization'].append(codebook_utilization)
                results['codebook_perplexity'].append(codebook_perplexity.item())
                results['wasserstein_distance'].append(wasserstein_distance.item())

                results_len = len(results['step'])
                data_frame = pd.DataFrame(data=results, index=range(1, results_len+1))
                data_frame.to_csv('uniform_wasserstein_quantizer_{}_results.csv'.format(args.mean), index_label='step')
                print('eval step:{}/{}, quant_error:{:.4f}, codebook_utilization:{:.4f}, codebook_perplexity:{:.4f}, wasserstein_distance:{:.4f}'.format(step, args.max_steps, quant_error.item(), codebook_utilization, codebook_perplexity.item(), wasserstein_distance.item()))
        
        elif args.vector_quantizer == "vanilla_quantizer":
            loss = quantizer.calc_commit_loss(z)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            if step == 1 or step%10 == 0:
                print('train step:{}/{}, commit loss:{:.4f}'.format(step, args.max_steps, loss.item()))
            if step == 1 or step%100 == 0:
                z = torch.randn(args.val_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
                quant_error, codebook_utilization, codebook_perplexity, wasserstein_distance = quantizer.calc_metrics(z)

                results['step'].append(step)
                results['quant_error'].append(quant_error.item())
                results['codebook_utilization'].append(codebook_utilization)
                results['codebook_perplexity'].append(codebook_perplexity.item())
                results['wasserstein_distance'].append(wasserstein_distance.item())

                results_len = len(results['step'])
                data_frame = pd.DataFrame(data=results, index=range(1, results_len+1))
                data_frame.to_csv('uniform_vanilla_quantizer_{}_results.csv'.format(args.mean), index_label='step')
                print('eval step:{}/{}, quant_error:{:.4f}, codebook_utilization:{:.4f}, codebook_perplexity:{:.4f}, wasserstein_distance:{:.4f}'.format(step, args.max_steps, quant_error.item(), codebook_utilization, codebook_perplexity.item(), wasserstein_distance.item()))
        
        elif args.vector_quantizer == "linear_quantizer":
            loss = quantizer.calc_commit_loss(z)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            if step == 1 or step%10 == 0:
                print('train step:{}/{}, commit loss:{:.4f}'.format(step, args.max_steps, loss.item()))
            if step == 1 or step%100 == 0:
                z = torch.randn(args.val_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
                quant_error, codebook_utilization, codebook_perplexity, wasserstein_loss = quantizer.calc_metrics(z)

                results['step'].append(step)
                results['quant_error'].append(quant_error.item())
                results['codebook_utilization'].append(codebook_utilization)
                results['codebook_perplexity'].append(codebook_perplexity.item())
                results['wasserstein_distance'].append(wasserstein_distance.item())

                results_len = len(results['step'])
                data_frame = pd.DataFrame(data=results, index=range(1, results_len+1))
                data_frame.to_csv('uniform_linear_quantizer_{}_results.csv'.format(args.mean), index_label='step')
                print('eval step:{}/{}, quant_error:{:.4f}, codebook_utilization:{:.4f}, codebook_perplexity:{:.4f}, wasserstein_loss:{:.4f}'.format(step, args.max_steps, quant_error.item(), codebook_utilization, codebook_perplexity.item(), wasserstein_loss.item()))

        elif args.vector_quantizer == "ema_quantizer":
            quant_error = quantizer.ema_update(z)
            if step == 1 or step%10 == 0:
                print('train step:{}/{}, quant_error:{:.4f}'.format(step, args.max_steps, quant_error.item()))
            if step == 1 or step%100 == 0:
                z = torch.randn(args.val_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
                quant_error, codebook_utilization, codebook_perplexity, wasserstein_distance = quantizer.calc_metrics(z)

                results['step'].append(step)
                results['quant_error'].append(quant_error.item())
                results['codebook_utilization'].append(codebook_utilization)
                results['codebook_perplexity'].append(codebook_perplexity.item())
                results['wasserstein_distance'].append(wasserstein_distance.item())

                results_len = len(results['step'])
                data_frame = pd.DataFrame(data=results, index=range(1, results_len+1))
                data_frame.to_csv('uniform_ema_quantizer_{}_results.csv'.format(args.mean), index_label='step')
                print('eval step:{}/{}, quant_error:{:.4f}, codebook_utilization:{:.4f}, codebook_perplexity:{:.4f}, wasserstein_distance:{:.4f}'.format(step, args.max_steps, quant_error.item(), codebook_utilization, codebook_perplexity.item(), wasserstein_distance.item()))
            
        elif args.vector_quantizer == "online_clustering":
            loss = quantizer.calc_commit_loss(z)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            quant_error = quantizer.online_update(z)
            if step == 1 or step%10 == 0:
                print('train step:{}/{}, quant_error:{:.4f}'.format(step, args.max_steps, quant_error.item()))
            if step == 1 or step%100 == 0:
                z = torch.randn(args.val_samples, args.codebook_dim).uniform_(-1.0, 1.0).cuda() + args.mean
                quant_error, codebook_utilization, codebook_perplexity, wasserstein_distance = quantizer.calc_metrics(z)

                results['step'].append(step)
                results['quant_error'].append(quant_error.item())
                results['codebook_utilization'].append(codebook_utilization)
                results['codebook_perplexity'].append(codebook_perplexity.item())
                results['wasserstein_distance'].append(wasserstein_distance.item())

                results_len = len(results['step'])
                data_frame = pd.DataFrame(data=results, index=range(1, results_len+1))
                data_frame.to_csv('uniform_online_quantizer_{}_results.csv'.format(args.mean), index_label='step')
                print('eval step:{}/{}, quant_error:{:.4f}, codebook_utilization:{:.4f}, codebook_perplexity:{:.4f}, wasserstein_distance:{:.4f}'.format(step, args.max_steps, quant_error.item(), codebook_utilization, codebook_perplexity.item(), wasserstein_distance.item()))

if __name__ == '__main__':
    args = parse_arg()
    dict_args = vars(args)
    for k, v in zip(dict_args.keys(), dict_args.values()):
        print("{0}: {1}".format(k, v))
    main_worker(args)