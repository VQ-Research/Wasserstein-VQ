import sys
import os
from collections import defaultdict
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from torch import nn, optim
import math
from omegaconf import OmegaConf
import yaml
from torch import inf

def adjust_learning_rate(optimizer, step, args, min_lr_constant=5):
    if step < args.warmup_iters:
        lr = args.lr * (step / args.warmup_iters)
    elif step > args.decay_iters:
        lr = args.lr/min_lr_constant
    else:
        decay_ratio = float(step - args.warmup_iters) / (args.decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        lr = args.lr / min_lr_constant + coeff * (args.lr - args.lr / min_lr_constant)

    optimizer.param_groups[0]["lr"] = lr
    return lr

class Logger(object):
    def __init__(self, saver_dir, saver_name_pre):
        self.terminal = sys.stdout
        output_file = os.path.join(saver_dir, saver_name_pre+"-record.log") 
        self.log = open(output_file, "w")

    def write(self, message):
        print(message, end="", file=self.terminal, flush=True)
        print(message, end="", file=self.log, flush=True)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list[v]
            else:
                pack[k] = v
        return pack

class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                try:
                    self.losses[key].append(val.item())
                except:
                    self.losses[key].append(val)
                    
    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            else:
                avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
                str_losses.append("{} {:.4f},".format(key, avg_loss))
        if prefix:
            return "{} {}".format(prefix, " ".join(str_losses))
        else:
            return "{}".format(" ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses) 

