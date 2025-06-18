import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

from model.vanilla_quantizer import Vanilla_Quantizer
from model.wasserstein_quantizer import Wasserstein_Quantizer
from model.ema_quantizer import EMA_Quantizer
from model.online_quantizer import Online_Quantizer
from model.encoder_decoder import Encoder, Decoder
from utils.util import Pack

class VQVAE(nn.Module):
    def __init__(self, args):
        super(VQVAE, self).__init__()
        self.args = args
        self.codebook_dim = args.codebook_dim

        ### encoder and decoder
        if args.factor == 16:
            self.encoder = Encoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=64, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)
            self.decoder = Decoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=64, ch_mult=[1, 1, 2, 2, 4], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)
        elif args.factor == 8:
            self.encoder = Encoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=64, ch_mult=[1, 1, 2, 2], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
            self.decoder = Decoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=64, ch_mult=[1, 1, 2, 2], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
        elif args.factor == 4:
            self.encoder = Encoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=64, ch_mult=[1, 1, 2], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
            self.decoder = Decoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=64, ch_mult=[1, 1, 2], num_res_blocks=2, attn_resolutions=[], dropout=0.0)
            
        ### quantizer
        if args.quantizer_name == 'wasserstein_quantizer':
            self.quantizer = Wasserstein_Quantizer(args)
        elif args.quantizer_name == 'vanilla_quantizer':
            self.quantizer = Vanilla_Quantizer(args)
        elif args.quantizer_name == 'ema_quantizer':
            self.quantizer = EMA_Quantizer(args)
        elif args.quantizer_name == 'online_quantizer':
            self.quantizer = Online_Quantizer(args)

        ### pre_quant_cov and post_quant_conv
        self.pre_quant_proj = torch.nn.Conv2d(args.z_channels, args.codebook_dim, kernel_size=1)
        self.post_quant_proj = torch.nn.Conv2d(args.codebook_dim, args.z_channels, kernel_size=1)

    def encode(self, x):
        ### this function extracts visual tokens from the vanilla images
        z = self.encoder(x)
        z = self.pre_quant_proj(z)
        indices = self.quantizer.obtain_embedding_id(x)
        return indices

    def decode(self, indices):
        ### This function generates new images when generated visual tokens (or say indices) are given 
        z = self.quantizer.obtain_codebook_entry(indices) ## (b,h,w,c)
        z = rearrange(z, 'b h w c -> b c h w')
        z = self.post_quant_proj(z)
        x = self.decoder(z).clamp_(-1, 1)
        return x

    def forward(self, x):
        ## encoder
        z = self.encoder(x)
        z = self.pre_quant_proj(z)

        ## quantizer 
        if self.args.quantizer_name == 'wasserstein_quantizer':
            z_q, vq_loss, wasserstein_loss, quant_error, codebook_utilization, codebook_perplexity = self.quantizer(z)
        else:
            z_q, vq_loss, quant_error, codebook_utilization, codebook_perplexity = self.quantizer(z)
        
        ## decoder
        z = self.post_quant_proj(z_q)
        x_rec = self.decoder(z)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        loss = rec_loss + vq_loss

        if self.args.quantizer_name == 'wasserstein_quantizer':
            loss_pack = Pack(loss=loss, vq_loss=vq_loss, rec_loss=rec_loss, wasserstein_loss=wasserstein_loss, quant_error=quant_error, codebook_utilization=codebook_utilization, codebook_perplexity=codebook_perplexity)
        else:
            loss_pack = Pack(loss=loss, vq_loss=vq_loss, rec_loss=rec_loss, quant_error=quant_error, codebook_utilization=codebook_utilization, codebook_perplexity=codebook_perplexity)
        return loss, loss_pack
    
    def collect_eval_info(self, x):
        ## encoder
        z = self.encoder(x)
        z = self.pre_quant_proj(z)

        ## quantizer 
        if self.args.quantizer_name == 'wasserstein_quantizer':
            z_q, wasserstein_loss, quant_error, histogram = self.quantizer.collect_eval_info(z)
        else:
            z_q, quant_error, histogram = self.quantizer.collect_eval_info(z)
        
        ## decoder
        z = self.post_quant_proj(z_q)
        x_rec = self.decoder(z).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        
        if self.args.quantizer_name == 'wasserstein_quantizer':
            return x_rec, rec_loss, wasserstein_loss, quant_error, histogram
        else:
            return x_rec, rec_loss, quant_error, histogram

    def obtain_feature(self, x):
        z = self.encoder(x)
        z = self.pre_quant_proj(z)

        z = rearrange(z, 'b c h w -> b h w c')
        z_flat = z.reshape(-1, self.codebook_dim)
        return z_flat


