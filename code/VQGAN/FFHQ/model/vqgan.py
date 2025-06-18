import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from model.wasserstein_quantizer import Wasserstein_Quantizer
from model.encoder_decoder import Encoder, Decoder
from model.lpips import LPIPS
from model.discriminator import PatchGANDiscriminator
from utils.util import Pack

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.args = args
        self.disc_adaptive_weight = False

        ### encoder and decoder
        if args.factor == 16:
            self.encoder = Encoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=128, ch_mult=[1, 1, 2, 2, 2], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)
            self.decoder = Decoder(double_z=False, z_channels=args.z_channels, resolution=args.resolution, in_channels=args.channels,
                             out_ch=args.channels, ch=128, ch_mult=[1, 1, 2, 2, 2], num_res_blocks=2, attn_resolutions=[16], dropout=0.0)            

        ### pre_quant_cov and post_quant_conv
        self.pre_quant_proj = torch.nn.Conv2d(args.z_channels, args.codebook_dim, kernel_size=1)
        self.post_quant_proj = torch.nn.Conv2d(args.codebook_dim, args.z_channels, kernel_size=1)
        self.quantizer = Wasserstein_Quantizer(args)

        self.lpips = LPIPS().eval()
        self.discriminator = PatchGANDiscriminator()

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

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def hinge_gen_loss(self, logit_fake):
        return -torch.mean(logit_fake)

    def adopt_weight(self, weight, global_epoch, threshold=0, value=0.):
        if global_epoch < threshold:
            weight = value
        return weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def vqvae(self, x):
        ## encoder
        z = self.encoder(x)
        z = self.pre_quant_proj(z)
        z_q, vq_loss, wasserstein_loss, quant_error, codebook_utilization, codebook_perplexity = self.quantizer(z)

        ## decoder
        z = self.post_quant_proj(z_q)
        x_rec = self.decoder(z)

        info_pack = Pack(vq_loss=vq_loss, wasserstein_loss=wasserstein_loss, quant_error=quant_error, codebook_utilization=codebook_utilization, codebook_perplexity=codebook_perplexity)
        return x_rec, vq_loss, info_pack

    def vqgan(self, x, x_rec, vq_loss, global_epoch, optimizer_idx, last_layer=None): 
        ## generator update
        if optimizer_idx == 0:
            rec_loss = torch.mean(torch.abs(x.contiguous() - x_rec.contiguous()))
            lpips_loss = torch.mean(self.lpips(x.contiguous(), x_rec.contiguous()))

            logits_fake = self.discriminator(x_rec.contiguous())
            g_loss = self.hinge_gen_loss(logits_fake)

            if self.disc_adaptive_weight:
                null_loss = rec_loss + lpips_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, g_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1.0
            disc_weight = self.adopt_weight(self.args.rate_d, global_epoch, threshold=self.args.disc_epoch)

            gen_loss = rec_loss + vq_loss + lpips_loss + disc_weight * g_loss
            loss_pack = Pack(gen_loss=gen_loss, rec_loss=rec_loss, lpips_loss=lpips_loss, g_loss=g_loss, disc_weight=disc_weight, disc_adaptive_weight=disc_adaptive_weight)
            return gen_loss, loss_pack

        if optimizer_idx == 1:
            logits_real = self.discriminator(x.contiguous().detach())
            logits_fake = self.discriminator(x_rec.contiguous().detach())

            disc_weight = self.adopt_weight(self.args.rate_d*0.5, global_epoch, threshold=self.args.disc_epoch)
            d_loss = disc_weight * self.hinge_d_loss(logits_real, logits_fake)

            logits_real = logits_real.detach().mean()
            logits_fake = logits_fake.detach().mean()
            
            loss_pack = Pack(d_loss=d_loss, logits_real=logits_real, logits_fake=logits_fake)
            return d_loss, loss_pack
    
    def collect_eval_info(self, x):
        ## encoder
        z = self.encoder(x)
        z = self.pre_quant_proj(z)
        z_q, wasserstein_loss, quant_error, histogram = self.quantizer.collect_eval_info(z)
        
        ## decoder
        z = self.post_quant_proj(z_q)
        x_rec = self.decoder(z).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        
        return x_rec, rec_loss, wasserstein_loss, quant_error, histogram
       

    
