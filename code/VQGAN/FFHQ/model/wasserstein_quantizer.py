import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
import os

class Queue(nn.Module):
    def __init__(self, args):
        super(Queue, self).__init__()
        self.args = args
        self.queue_size = 65536
        self.register_buffer("queue", torch.randn(self.queue_size, args.codebook_dim)*0.01)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, key):
        batch_size = key.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        self.queue[ptr:ptr + batch_size, :] = key
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  

    @torch.no_grad()
    def obtain_feature_from_queue(self):
        return self.queue.detach().clone()

class Wasserstein_Quantizer(nn.Module):
    def __init__(self, args):
        super(Wasserstein_Quantizer, self).__init__()
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.beta = args.beta
        self.alpha = args.alpha
        self.gamma = args.gamma ##wasserstein loss coefficient

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.normal_(0, 0.01)
        self.embedding.weight.requires_grad = True
        self.queue = Queue(args)

    def calc_wasserstein_loss(self, z=None):
        if z==None:
            z = self.queue.obtain_feature_from_queue()

        N = z.size(0)
        D = z.size(1)

        c = self.embedding.weight
        std = c.std(dim=0).max().detach()

        # Normalize z and c
        z = z / (std + 1e-8)
        c = c / (std + 1e-8)

        z_mean = z.mean(0).detach()
        z_covariance = torch.cov(z.t()) + 1e-6 * torch.eye(D, device=z.device)  ##z_covariance = torch.mm((z - torch.mean(z, dim=0, keepdim=True)).t(), z - torch.mean(z, dim=0, keepdim=True))/(N-1)
        z_covariance = z_covariance.detach()

        ### compute the mean and covariance of codebook vectors
        c_mean = c.mean(0)
        c_covariance = torch.cov(c.t()) + 1e-6 * torch.eye(D, device=z.device)  ##c_covariance = torch.mm((c - torch.mean(c, dim=0, keepdim=True)).t(), c - torch.mean(c, dim=0, keepdim=True))/(self.codebook_size-1)
        
        ### calculation of part1
        part_mean =  torch.sum(torch.multiply(z_mean - c_mean, z_mean - c_mean))

        ### 1/2 covariance
        S1, Q1 = torch.linalg.eigh(z_covariance)
        sqrt_S1 = torch.diag(torch.sqrt(F.relu(S1)+ 1e-8))
        temp = torch.mm(Q1, sqrt_S1)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = torch.mm(temp, Q1.T)
        z_sqrt_covariance = torch.nan_to_num(z_sqrt_covariance, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = z_sqrt_covariance.detach()
        
        ### 1/2 covariance
        temp = torch.mm(z_sqrt_covariance, c_covariance)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

        covariance = torch.mm(temp, z_sqrt_covariance)
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

        S2, Q2 = torch.linalg.eigh(covariance)
        sqrt_S2 = torch.sqrt(F.relu(S2)+ 1e-8)

        #############calculation of part2
        part_covariance = F.relu(torch.trace(z_covariance.detach() + c_covariance) - 2.0 * sqrt_S2.sum())
        wasserstein_loss = torch.sqrt(part_mean + part_covariance + 1e-10)
        return wasserstein_loss
        
    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        with torch.no_grad():
            self.queue.dequeue_and_enqueue(z_flattened.detach())
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight.data) # 'n d -> d n'

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        ## The only difference to the Vanilla Quantizer
        wasserstein_loss = self.calc_wasserstein_loss()
        loss = self.beta * F.mse_loss(z_q.detach(), z) + self.alpha * F.mse_loss(z_q, z.detach()) + self.gamma * wasserstein_loss 
        # preserve gradients
        z_q = z + (z_q - z).detach()

        ## Criterion Triple defined in the paper
        quant_error = (z_q.detach()-z.detach()).square().sum(3).mean()

        histogram = token.bincount(minlength=self.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        codebook_utilization = codebook_usage_counts.item() / self.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, wasserstein_loss, quant_error, codebook_utilization, codebook_perplexity
    
    def collect_eval_info(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)

        wasserstein_loss = self.calc_wasserstein_loss(z_flattened.detach())
        # distances from z to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        quant_error = (z_q.detach()-z.detach()).square().sum(3).mean()

        histogram = token.bincount(minlength=self.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, wasserstein_loss, quant_error, histogram

    def obtain_embedding_id(self, z):
        b, c, h, w = z.shape
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)

        # distances from z to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        return token.view(b, h, w)
    
    def obtain_codebook_entry(self, indices):
        return self.embedding(indices)  ## (b,h,w,c)