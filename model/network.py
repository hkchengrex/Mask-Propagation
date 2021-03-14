"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x

class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, mk, mv, qk, qv):
        B, CK, T, H, W = mk.shape
        _, CV, _, _, _ = mv.shape

        mi = mk.view(B, CK, T*H*W) 
        mi = torch.transpose(mi, 1, 2) # B * THW * CK
 
        qi = qk.view(B, CK, H*W) / math.sqrt(CK)  # B * CK * HW
 
        affinity = torch.bmm(mi, qi)  # B, THW, HW
        affinity = F.softmax(affinity, dim=1)  # B, THW, HW

        mv = mv.view(B, CV, T*H*W) 
        mem = torch.bmm(mv, affinity) # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class PropagationNetwork(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        if single_object:
            self.mask_rgb_encoder = MaskRGBEncoderSO() 
        else:
            self.mask_rgb_encoder = MaskRGBEncoder() 
        self.rgb_encoder = RGBEncoder()

        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True),
            prob
        ], 1).clamp(1e-7, 1-1e-7)
        logits = torch.log((new_prob /(1-new_prob)))
        return logits

    def memorize(self, frame, mask, other_mask=None): 
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.mask_rgb_encoder(frame, mask)
        else:
            f16 = self.mask_rgb_encoder(frame, mask, other_mask)
        k16, v16 = self.kv_m_f16(f16)
        return k16.unsqueeze(2), v16.unsqueeze(2) # B*C*T*H*W

    def segment(self, frame, keys, values, selector=None): 
        b, k = keys.shape[:2]

        f16, f8, f4 = self.rgb_encoder(frame)
        k16, v16 = self.kv_q_f16(f16)  
        
        if self.single_object:
            logits = self.decoder(self.memory(keys, values, k16, v16), f8, f4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                self.decoder(self.memory(keys[:,0], values[:,0], k16, v16), f8, f4),
                self.decoder(self.memory(keys[:,1], values[:,1], k16, v16), f8, f4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        return logits, prob

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)
        else:
            return self.memorize(*args, **kwargs)


