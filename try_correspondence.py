import os
from os import path
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import cv2

from model.corr_network import CorrespondenceNetwork
from dataset.range_transform import im_normalization
from util.tensor_util import pad_divide_by, unpad


parser = ArgumentParser()
parser.add_argument('--src_image', default='./examples/00015.jpg')
parser.add_argument('--tar_image', default='./examples/00020.jpg')
parser.add_argument('--model', default='saves/propagation_model.pth')
args = parser.parse_args()

# loading a pretrained propagation network as correspondence network
prop_saved = torch.load(args.model)
corr_net = CorrespondenceNetwork().cuda().eval()
corr_net.load_state_dict(prop_saved, strict=False)
torch.set_grad_enabled(False)

# remember the normalization!
im_transform = transforms.Compose([
    transforms.ToTensor(),
    im_normalization
])

def image_to_tensor(im):
    im = im_transform(im).unsqueeze(0)
    return im.cuda()

# Reading stuff
src_image = Image.open(args.src_image).convert('RGB')
tar_image = Image.open(args.tar_image).convert('RGB')
src_im_th = image_to_tensor(src_image)
tar_im_th = image_to_tensor(tar_image)

""" 
Compute W
"""
# Inputs need to have dimensions as multiples of 16
src_im_th, pads = pad_divide_by(src_im_th, 16)
tar_im_th, _ = pad_divide_by(tar_im_th, 16)

# Mask input is not crucial to getting a good correspondence
# we are just using an empty mask here
b, _, h, w = src_im_th.shape
empty_mask = torch.zeros((b, 1, h, w), device=src_im_th.device)

# We can precompute the affinity matrix (H/16 * W/16) * (H/16 * W/16)
# 16 is the encoder stride
qk16 = corr_net.get_query_key(tar_im_th)
mk16 = corr_net.get_mem_key(src_im_th, empty_mask, empty_mask)
W = corr_net.get_W(mk16, qk16)

# Generate the transfer mask
# This mask is considered as our "feature" to be transferred using the affinity matrix
# A feature vectors can also be used (i.e. channel size > 1)
nh, nw = h//16, w//16
transfer_mask = torch.zeros((b, 1, nh, nw), device=src_im_th.device)

def match(W, transfer_feat):
    # This is mostly just torch.bmm(features, affinity)
    transferred = corr_net.transfer(W, transfer_feat)
    # Upsample 16 stride image to original size
    transferred = F.interpolate(transferred, scale_factor=16, mode='bilinear', align_corners=False)
    # Remove padding introduced at the beginning
    transferred = unpad(transferred, pads)
    return transferred

"""
Just visualization and interaction stuff
"""
src_image = np.array(src_image)
src_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
tar_image = np.array(tar_image)
tar_image = cv2.cvtColor(tar_image, cv2.COLOR_RGB2BGR)

changed = True
click = (0, 0)
def mouse_callback(event, x, y, flags, param):
    global changed, click
    # Changing modes
    if event == cv2.EVENT_LBUTTONDOWN:
        changed = True
        click = (x, y)
        transfer_mask.zero_()
        transfer_mask[0,0,y//16,x//16] = 1

def comp_binary(image, mask):
    # Increase the brightness a bit
    mask = (mask*2).clip(0, 1)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:,:,2] = 255
    if len(mask.shape) == 2:
        mask = mask[:,:,None]
    image_dim = image*(1-mask)*0.7 + mask*image*0.3
    comp = (image_dim + color_mask*mask*0.7).astype(np.uint8)

    return comp

# OpenCV setup
cv2.namedWindow('Source')
cv2.namedWindow('Target')
cv2.setMouseCallback('Source', mouse_callback)

while 1:
    if changed:
        click_map_vis = F.interpolate(transfer_mask, scale_factor=16, mode='bilinear', align_corners=False)
        click_map_vis = unpad(click_map_vis, pads)
        click_map_vis = click_map_vis[0,0].cpu().numpy()
        attn_map = match(W, transfer_mask)
        attn_map = attn_map/(attn_map.max()+1e-6)
        # Scaling for visualization
        attn_map = attn_map[0,0].cpu().numpy()

        tar_vis = comp_binary(tar_image, attn_map)
        src_vis = comp_binary(src_image, click_map_vis)

        cv2.imshow('Source', src_vis)
        cv2.imshow('Target', tar_vis)
        changed = False

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
