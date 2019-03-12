#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Date: 2019/2/28
Author: Xu Yucheng
Abstract: Code for building ting-yolo-v1 in pytorch
'''
from __future__ import division

import os
import sys
import time
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from build_model import *

class Darknet(nn.Module):
    def __init__(self, cfg_file):
        super(Darknet, self).__init__()
        self.blocks = _parse_cfg(cfg_file)
        self.net_info, self.module_list = _create_module(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks
        outputs = {}

        #write表示我们是否遇到第一个检测。write=0，则收集器尚未初始化，write=1，则收集器已经初始化，我们只需要将检测图与收集器级联起来即可。
        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type = 'convolutional' or module_type == 'maxpool':
                x = self.module_list[i](x)
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                input_dim = int(self.net_info["height"])