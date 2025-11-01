import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_quantizer import VanillaVectorQuantizer
from models.ema_quantizer import EMAVectorQuantizer
from models.online_quantizer import OnlineVectorQuantizer
from models.wasserstein_quantizer import WassersteinVectorQuantizer
from models.mmd_quantizer import MMDVectorQuantizer
from models.encoder_decoder import Encoder, Decoder, Normalize
from utils.util import Pack
from safetensors.torch import load_file
from models.lpips import LPIPS

class VQModel(nn.Module):
    def __init__(self, args):
        super(VQModel, self).__init__()
        self.args = args
        ddconfig = dict(
            dropout=0, ch=160, z_channels=32,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   
            using_sa=True, using_mid_sa=True,                          
        )
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quant_conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.post_quant_conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)

        if args.VQ == "vanilla_vq":
            self.quantizer1 = VanillaVectorQuantizer(args)
            self.quantizer2 = VanillaVectorQuantizer(args)
        elif args.VQ == "ema_vq":
            self.quantizer1 = EMAVectorQuantizer(args)
            self.quantizer2 = EMAVectorQuantizer(args)
        elif args.VQ == "online_vq":
            self.quantizer1 = OnlineVectorQuantizer(args)
            self.quantizer2 = OnlineVectorQuantizer(args)
        elif args.VQ == "wasserstein_vq":
            self.quantizer1 = WassersteinVectorQuantizer(args)
            self.quantizer2 = WassersteinVectorQuantizer(args)
        elif args.VQ == "mmd_vq":
            self.quantizer1 = MMDVectorQuantizer(args)
            self.quantizer2 = MMDVectorQuantizer(args)

        self.projector_in = nn.Sequential(
                nn.Conv2d(32, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 32, kernel_size=3, padding=1),
            )

        self.projector_out = nn.Sequential(
                nn.Conv2d(32, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                Normalize(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 32, kernel_size=3, padding=1),
            )

        if args.stage == "transplant":
            self.perceptual_loss = LPIPS().eval()
            pretrain_dict = torch.load(args.pretrained_tokenizer, map_location='cpu', weights_only=False)
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quant_conv.')}
            post_quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('post_quant_conv.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quant_conv_dict = {k.replace('quant_conv.', '', 1): v for k, v in quant_conv_dict.items()}
            post_quant_conv_dict = {k.replace('post_quant_conv.', '', 1): v for k, v in post_quant_conv_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quant_conv.load_state_dict(quant_conv_dict, strict=True)
            self.post_quant_conv.load_state_dict(post_quant_conv_dict, strict=True)

            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False
            for param in self.projector_in.parameters():
                param.requires_grad = True
            for param in self.quantizer1.parameters():
                param.requires_grad = True
            for param in self.quantizer2.parameters():
                param.requires_grad = True
            for param in self.projector_out.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()
            self.quant_conv.eval()
            self.post_quant_conv.eval()

        if args.stage == "refinement":
            checkpoint_dir = os.path.join(os.path.join(args.init_checkpoint_dir, "Transplant"), args.dataset_name)
            checkpoint_name = args.checkpoint_name
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            pretrain_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quant_conv.')}
            post_quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('post_quant_conv.')}
            quantizer1_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
            quantizer2_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
            projector_in_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_in.')}
            projector_out_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_out.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quant_conv_dict = {k.replace('quant_conv.', '', 1): v for k, v in quant_conv_dict.items()}
            post_quant_conv_dict = {k.replace('post_quant_conv.', '', 1): v for k, v in post_quant_conv_dict.items()}
            quantizer1_dict = {k.replace('quantizer1.', '', 1): v for k, v in quantizer1_dict.items()}
            quantizer2_dict = {k.replace('quantizer2.', '', 1): v for k, v in quantizer2_dict.items()}
            projector_in_dict = {k.replace('projector_in.', '', 1): v for k, v in projector_in_dict.items()}
            projector_out_dict = {k.replace('projector_out.', '', 1): v for k, v in projector_out_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quant_conv.load_state_dict(quant_conv_dict, strict=True)
            self.post_quant_conv.load_state_dict(post_quant_conv_dict, strict=True)
            self.quantizer1.load_state_dict(quantizer1_dict, strict=True)
            self.quantizer2.load_state_dict(quantizer2_dict, strict=True)
            self.projector_in.load_state_dict(projector_in_dict, strict=True)
            self.projector_out.load_state_dict(projector_out_dict, strict=True)

            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False          
            for param in self.quantizer1.parameters():
                param.requires_grad = False
            for param in self.quantizer2.parameters():
                param.requires_grad = False
            for param in self.projector_in.parameters():
                param.requires_grad = False
            for param in self.projector_out.parameters():
                param.requires_grad = True
            for param in self.post_quant_conv.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.encoder.eval()
            self.quant_conv.eval()
            self.quantizer1.eval()
            self.quantizer2.eval()
            self.projector_in.eval()

    def transplant(self, x):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            ze = self.encoder(x)
            z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
        z_q_1, vq_loss_1, utilization_1, perplexity_1 = self.quantizer1(z_p_1)
        z_q_2, vq_loss_2, utilization_2, perplexity_2 = self.quantizer2(z_p_2)
        z_q = torch.cat((z_q_1, z_q_2), dim=1)
        z_q = z_q + self.projector_out(z_q)
        vq_loss = (vq_loss_1 + vq_loss_2) * 0.5

        loss = F.mse_loss(z_q, z_obj.detach())
        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q)

        p_loss = self.perceptual_loss(x.contiguous(), x_rec.contiguous())
        p_loss = torch.mean(p_loss)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        transplant_loss = 5.0 * rec_loss + p_loss + loss + vq_loss

        utilization = (utilization_1 + utilization_2) * 0.5
        perplexity = (perplexity_1 + perplexity_2) * 0.5
        return  transplant_loss, rec_loss, p_loss, quant_error, utilization, perplexity

    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            ze = self.encoder(x)
            z_obj = self.quant_conv(ze)

            z_p = z_obj + self.projector_in(z_obj)
            z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
            z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        
        z_q = z_q + self.projector_out(z_q)
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q)
        return x_rec

    def collect_eval_info_transplant(self, x):
        ze = self.encoder(x)
        z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
        z_q_1, histogram_1 = self.quantizer1.collect_eval_info(z_p_1)
        z_q_2, histogram_2 = self.quantizer2.collect_eval_info(z_p_2)
        z_q = torch.cat((z_q_1, z_q_2), dim=1)
        z_q = z_q + self.projector_out(z_q)

        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, quant_error, histogram_1, histogram_2

    def collect_eval_info_refinement(self, x):
        ze = self.encoder(x)
        z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
        z_q_1, _ = self.quantizer1.collect_eval_info(z_p_1)
        z_q_2, _ = self.quantizer2.collect_eval_info(z_p_2)
        z_q = torch.cat((z_q_1, z_q_2), dim=1)
        z_q = z_q + self.projector_out(z_q)

        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
        
    def reconstruction(self, x):
        ze = self.encoder(x)
        z_obj = self.quant_conv(ze)

        z_p = z_obj + self.projector_in(z_obj)
        z_p_1, z_p_2 = torch.chunk(z_p, 2, dim=1)
        z_q_1 = self.quantizer1.collect_reconstruction(z_p_1)
        z_q_2 = self.quantizer2.collect_reconstruction(z_p_2)
        z_q = torch.cat((z_q_1, z_q_2), dim=1)
        z_q = z_q + self.projector_out(z_q)

        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        return x_rec

