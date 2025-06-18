"""Adapted from taming transformers: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from torch import distributed as tdist

class EmbeddingEMA(nn.Module):
    def __init__(self, codebook_size, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps

        weight = torch.randn(codebook_size, codebook_dim)/self.codebook_size
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(codebook_size), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   

### K-means
class EMA_Quantizer(nn.Module):
    def __init__(self, args, eps=1e-5):
        super(EMA_Quantizer, self).__init__()
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.decay = 0.8
        self.beta = args.beta
        self.embedding = EmbeddingEMA(self.codebook_size, self.codebook_dim, self.decay, eps)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight.data) # 'n d -> d n'

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        encodings = F.one_hot(token, self.codebook_size).type(z.dtype).detach()
        if self.training:
            #EMA cluster size
            encodings_sum = encodings.sum(0)            
            self.embedding.cluster_size_ema_update(encodings_sum)
            #EMA embedding average
            embed_sum = encodings.transpose(0,1) @ z_flattened            
            self.embedding.embed_avg_ema_update(embed_sum)
            #normalize embed_avg and update weight
            self.embedding.weight_update(self.codebook_size)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 

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
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
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

