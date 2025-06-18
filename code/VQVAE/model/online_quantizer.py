"""Adapted from Online Clustered Codebook: https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from torch import distributed as tdist

### K-means++ (closest online clustering)
class Online_Quantizer(nn.Module):
    def __init__(self, args):
        super(Online_Quantizer, self).__init__()
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.decay = 0.99
        self.beta = args.beta
        self.alpha = args.alpha

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 /self.codebook_size, 1.0/self.codebook_size)
        self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous() ## b c h w -> b h w c
        z_flattened = z.view(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + self.alpha * torch.mean((z_q - z.detach()) ** 2)
 
        # preserve gradients
        z_q = z + (z_q - z).detach()

        ############### Online clustering start
        ### vanilla code from online clustering
        ##onehot_probs = F.one_hot(token, self.codebook_size).type(z.dtype)
        ##avg_probs = torch.mean(onehot_probs, dim=0)

        ##our revision would be more efficient
        histogram = token.bincount(minlength=self.codebook_size).float()
        avg_probs = histogram/histogram.sum(0)
        self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)

        ## random feature
        sort_distance, indices = d.sort(dim=0)
        random_feat = z_flattened.detach()[indices[-1,:]]
        
        #indices = torch.argmin(d, dim=0)
        #random_feat = z_flattened.detach()[indices]
    
        decay = torch.exp(-(self.embed_prob * self.codebook_size * 10)/(1-self.decay) - 1e-3).unsqueeze(1).repeat(1, self.codebook_dim)
        self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
        ############### Online clustering end

        ## Criterion Triple defined in the paper
        quant_error = (z_q.detach()-z.detach()).square().sum(3).mean()

        codebook_usage_counts = (histogram > 0).float().sum()
        codebook_utilization = codebook_usage_counts.item() / self.codebook_size
        codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, quant_error, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)

        # distances from z to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        quant_error = (z_q.detach()-z.detach()).square().sum(3).mean()

        histogram = token.bincount(minlength=self.codebook_size).float()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, quant_error, histogram

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


