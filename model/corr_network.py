"""
corr_network.py - Correspondence version
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class AttentionMemory(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Mk, Qk): 
        B, CK, H, W = Mk.shape

        Mk = Mk.view(B, CK, H*W) 
        Mk = torch.transpose(Mk, 1, 2)  # B * HW * CK
 
        Qk = Qk.view(B, CK, H*W).expand(B, -1, -1) / math.sqrt(CK)  # B * CK * HW
 
        affinity = torch.bmm(Mk, Qk) # B * HW * HW
        affinity = F.softmax(affinity, dim=1)

        return affinity

    def readout(self, affinity, mv):
        B, CV, H, W = mv.shape
        mv = mv.flatten(start_dim=2)

        readout = torch.bmm(mv, affinity) # Weighted-sum B, CV, HW
        readout = readout.view(B, CV, H, W)

        return readout

class CorrespondenceNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_rgb_encoder = MaskRGBEncoder() 
        self.rgb_encoder = RGBEncoder() 

        self.kv_m_f16 = KeyValue(1024, keydim=128, valdim=512)
        self.kv_q_f16 = KeyValue(1024, keydim=128, valdim=512)

        self.attn_memory = AttentionMemory()

    def get_query_key(self, frame):
        f16, _, _ = self.rgb_encoder(frame)
        k16, _ = self.kv_q_f16(f16)  
        return k16

    def get_mem_key(self, frame, mask, mask2):
        f16 = self.mask_rgb_encoder(frame, mask, mask2)
        k16, _ = self.kv_m_f16(f16) 
        return k16

    def get_W(self, mk16, qk16):
        W = self.attn_memory(mk16, qk16)
        return W

    def transfer(self, W, val):
        return self.attn_memory.readout(W, val)
