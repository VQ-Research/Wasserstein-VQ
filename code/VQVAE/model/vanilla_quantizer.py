"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from torch import distributed as tdist

"""
Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
avoids costly matrix multiplications and allows for post-hoc remapping of indices.
"""
class Vanilla_Quantizer(nn.Module):
    def __init__(self, args):
        super(Vanilla_Quantizer, self).__init__()
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.beta = args.beta
        self.alpha = args.alpha

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 /self.codebook_size, 1.0/self.codebook_size)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        # compute loss for embedding (we fix the beta bug here)
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + self.alpha * torch.mean((z_q - z.detach()) ** 2)
 
        # preserve gradients
        z_q = z + (z_q - z).detach()

        ## Criterion Triple defined in the paper
        quant_error = (z_q.detach()-z.detach()).square().sum(3).mean()

        histogram = token.bincount(minlength=self.codebook_size).float()
        codebook_usage_counts = (histogram > 0).float().sum()
        codebook_utilization = codebook_usage_counts.item() / self.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
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

        


    