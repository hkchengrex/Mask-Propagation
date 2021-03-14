"""
YouTubeVOS has a label structure that is more complicated than DAVIS 
Labels might not appear on the first frame (there might be no labels at all in the first frame)
Labels might not even appear on the same frame (i.e. Object 0 at frame 10, and object 1 at frame 15)
0 does not mean background -- it is simply "no-label"
and object indices might not be in order, there are missing indices somewhere in the validation set

Dealing with these makes the logic a bit convoluted here
It is not necessarily hacky but do understand that it is not as straightforward as DAVIS

Validation set only.
"""


import os
from os import path
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from model.eval_network import PropagationNetwork
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from inference_core_yv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/propagation_model.pth')
parser.add_argument('--yv', default='../YouTube')
parser.add_argument('--output')
parser.add_argument('--split', default='valid')
parser.add_argument('--use_km', action='store_true')
parser.add_argument('--no_top', action='store_true')
args = parser.parse_args()

yv_path = args.yv
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Load our checkpoint
prop_saved = torch.load(args.model)
top_k = None if args.no_top else 50
if args.use_km:
    prop_model = PropagationNetwork(top_k=top_k, km=5.6).cuda().eval()
else:
    prop_model = PropagationNetwork(top_k=top_k, km=None).cuda().eval()
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):
    rgb = data['rgb']
    msk = data['gt'][0]
    info = data['info']
    name = info['name'][0]
    k = len(info['labels'][0])
    gt_obj = info['gt_obj']
    size = info['size']

    torch.cuda.synchronize()
    process_begin = time.time()

    # Frames with labels, but they are not exhaustively labeled
    frames_with_gt = sorted(list(gt_obj.keys()))

    processor = InferenceCore(prop_model, rgb, num_objects=k)
    # min_idx tells us the starting point of propagation
    # Propagating before there are labels is not useful
    min_idx = 99999
    for i, frame_idx in enumerate(frames_with_gt):
        min_idx = min(frame_idx, min_idx)
        # Note that there might be more than one label per frame
        obj_idx = gt_obj[frame_idx][0].tolist()
        # Map the possibly non-continuous labels into a continuous scheme
        obj_idx = [info['label_convert'][o].item() for o in obj_idx]

        # Append the background label
        with_bg_msk = torch.cat([
            1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
            msk[:,frame_idx],
        ], 0).cuda()

        # We perform propagation from the current frame to the next frame with label
        if i == len(frames_with_gt) - 1:
            processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
        else:
            processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

    # Do unpad -> upsample to original size (we made it 480p)
    out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
    for ti in range(processor.t):
        prob = processor.prob[:,ti]

        if processor.pad[2]+processor.pad[3] > 0:
            prob = prob[:,:,processor.pad[2]:-processor.pad[3],:]
        if processor.pad[0]+processor.pad[1] > 0:
            prob = prob[:,:,:,processor.pad[0]:-processor.pad[1]]

        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
        out_masks[ti] = torch.argmax(prob, dim=0)
    
    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

    # Remap the indices to the original domain
    idx_masks = np.zeros_like(out_masks)
    for i in range(1, k+1):
        backward_idx = info['label_backward'][i].item()
        idx_masks[out_masks==i] = backward_idx

    torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += (idx_masks.shape[0] - min_idx)

    # Save the results
    this_out_path = path.join(out_path, name)
    os.makedirs(this_out_path, exist_ok=True)
    for f in range(idx_masks.shape[0]):
        if f >= min_idx:
            img_E = Image.fromarray(idx_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

    del rgb
    del msk
    del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
