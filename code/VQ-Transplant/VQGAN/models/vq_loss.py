import torch
import torch.nn as nn
import torch.nn.functional as F
from models.lpips import LPIPS
from models.discriminators import PatchGANDiscriminator, StyleGANDiscriminator, PatchGANMaskBitDiscriminator, DinoDiscriminator
from utils.diff_aug import DiffAugment
import torch.distributed as tdist
from utils.util import Pack

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)

def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))

def adopt_weight(weight, cur_epoch, threshold=0, value=0.):
    if cur_epoch < threshold:
        weight = value
    return weight

class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg

class VQLoss(nn.Module):
    def __init__(self, args):
        super(VQLoss, self).__init__()
        self.args = args
        self.disc_adaptive_weight = True
        self.perceptual_loss = LPIPS().eval()
        self.lecam_ema = LeCAM_EMA()
        self.discriminator = DinoDiscriminator()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, inputs, reconstructions, optimizer_idx, cur_epoch, last_layer=None):
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous())
            
            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            # discriminator loss
            reconstructions = DiffAugment(reconstructions.contiguous(), policy='color,translation,cutout_0.2', prob=0.5)
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = hinge_gen_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = rec_loss + p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, g_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1.

            gen_loss = rec_loss + p_loss + disc_adaptive_weight * g_loss
            disc_weight = adopt_weight(self.args.disc_weight, cur_epoch, threshold=self.args.disc_epoch)
            loss_pack = Pack(gen_loss=gen_loss, rec_loss=rec_loss, lpips_loss=p_loss, g_loss=g_loss, disc_weight=disc_weight, disc_adaptive_weight=disc_adaptive_weight)
            return gen_loss, loss_pack

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
            logits_fake = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))

            disc_weight = adopt_weight(self.args.disc_weight, cur_epoch, threshold=self.args.disc_epoch)

            self.lecam_ema.update(logits_real, logits_fake)
            lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
            adversarial_loss = hinge_d_loss(logits_real, logits_fake)
            d_loss = disc_weight * (lecam_loss * self.args.lecam_loss_weight + adversarial_loss)

            logits_real_s = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
            logits_fake_s = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
            disc_cr_loss_weight = self.args.disc_cr_loss_weight if cur_epoch >= self.args.disc_epoch else 0.0
            d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
            d_loss += d_cr

            logits_real = logits_real.detach().mean()
            logits_fake = logits_fake.detach().mean()

            loss_pack = Pack(d_loss=d_loss, logits_real=logits_real, logits_fake=logits_fake)
            return d_loss, loss_pack