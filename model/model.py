"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

from model.network import PropagationNetwork
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs


class PropagationModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        self.PNet = nn.parallel.DistributedDataParallel(
            PropagationNetwork(self.single_object).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.PNet.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb']
        Ms = data['gt']
        
        if self.single_object:
            key_k, key_v = self.PNet(Fs[:,0], Ms[:,0])
            prev_logits, prev_mask = self.PNet(Fs[:,1], key_k, key_v)
            prev_k, prev_v = self.PNet(Fs[:,1], prev_mask)

            keys = torch.cat([key_k, prev_k], 2)
            values = torch.cat([key_v, prev_v], 2)
            this_logits, this_mask = self.PNet(Fs[:,2], keys, values)

            out['mask_1'] = prev_mask
            out['mask_2'] = this_mask
            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits
        else:
            sec_Ms = data['sec_gt']
            selector = data['selector']

            key_k1, key_v1 = self.PNet(Fs[:,0], Ms[:,0], sec_Ms[:,0])
            key_k2, key_v2 = self.PNet(Fs[:,0], sec_Ms[:,0], Ms[:,0])
            key_k = torch.stack([key_k1, key_k2], 1)
            key_v = torch.stack([key_v1, key_v2], 1)

            prev_logits, prev_mask = self.PNet(Fs[:,1], key_k, key_v, selector)
            
            prev_k1, prev_v1 = self.PNet(Fs[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
            prev_k2, prev_v2 = self.PNet(Fs[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
            prev_k = torch.stack([prev_k1, prev_k2], 1)
            prev_v = torch.stack([prev_v1, prev_v2], 1)

            keys = torch.cat([key_k, prev_k], 3)
            values = torch.cat([key_v, prev_v], 3)

            this_logits, this_mask = self.PNet(Fs[:,2], keys, values, selector)

            out['mask_1'] = prev_mask[:,0:1]
            out['mask_2'] = this_mask[:,0:1]
            out['sec_mask_1'] = prev_mask[:,1:2]
            out['sec_mask_2'] = this_mask[:,1:2]

            out['logits_1'] = prev_logits
            out['logits_2'] = this_logits

        if self._do_log or self._is_train:
            losses = self.loss_computer.compute({**data, **out}, it)

            # Logging
            if self._do_log:
                self.integrator.add_dict(losses)
                if self._is_train:
                    if it % self.save_im_interval == 0 and it != 0:
                        if self.logger is not None:
                            images = {**data, **out}
                            size = (384, 384)
                            self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

        if self._is_train:
            if (it) % self.report_interval == 0 and it != 0:
                if self.logger is not None:
                    self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                    self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                self.last_time = time.time()
                self.train_integrator.finalize('train', it)
                self.train_integrator.reset_except_hooks()

            if it % self.save_model_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save(it)

            # Backward pass
            for param_group in self.optimizer.param_groups:
                for p in param_group['params']:
                    p.grad = None
            losses['total_loss'].backward() 
            self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.PNet.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.PNet.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.PNet.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'mask_rgb_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.PNet.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.PNet.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.PNet.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.PNet.eval()
        return self

