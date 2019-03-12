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

IMAGE_SIZE = 224
PATH_TO_CFG = '../cfg/yolov1-tiny.cfg'

def _parse_cfg(path=PATH_TO_CFG):
    """
        read yolov1-tiny.cfg
        Returns a list of blocks. Each blocks describes a block in the neural
        network to be built. Block is represented as a dictionary in the list
    """
    cfg_file = open(path, 'r')
    lines = cfg_file.read().split('\n')           # Store lines in a list
    lines = [x for x in lines if len(x) > 0]      # Get rid of the empty lines
    lines = [x for x in lines if x[0] != '#']     # Get rid of the comments
    lines = [x.rstrip().lstrip() for x in lines]  # Get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        # Marks the start of a new block
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            # Get the type of current block
            # [convolutional]
            block["type"] = line[1:-1].rstrip()
        else:
            # Get the parameters of the current block 
            # filters=16
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks

def _create_module(blocks):
    """
        blocks: the return of _parse_cfg
        use the blocks to build layers in a pytorch model
    """
    # Store [net] block in .cfg as net infomation
    net_info = blocks[0]

    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # Check the type of current block
        # Create a new module for current block
        # Append the module to module_list

        # If the current block is conv layer
        if x["type"] == 'convolutional':
            # Get the parameters in this layer
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Add a convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            prev_filters = filters
            output_filters.append(filters)

            # Add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            
            # Check the activation
            # YOLO uses a Leaky ReLU
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
                
        elif x['type'] == 'maxpool':
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
            module.add_module("maxpool_{0}".format(index), maxpool)
            

        
        # If the current block is upsampling layer
        # NOT included in yolov1-tiny
        elif x['type'] == 'upsamping':
            stride = int(x['stride'])
            upsamping = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module("upsampling_{0}".format(index), upsamping)
        
        # If the current is a route layer
        # NOT included in yolov1-tiny
        #    route层值得一些解释。 它具有可以具有一个或两个值的属性层。 
        #    当属性只有一个值时，它会输出由该值索引的网络层的特征图。 在我们的示例中，它是−4，因此这个层将从Route层向后输出第4层的特征图。
        #    当图层有两个值时，它会返回由其值所索引的图层的连接特征图。 在我们的例子中，它是−1,61，并且该图层将输出来自上一层（-1）和第61层的特征图，并沿深度的维度连接。
        
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')

            # Start of a route
            start = int(x['layers'][0])
            # End, if there exists one
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            
            # Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # If the current is Shortcut layer
        # NOT included in yolov1-
        # Shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut =  EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)    
        module_list.append(module)
    return net_info, module_list       

if __name__ == '__main__':
    blocks = _parse_cfg()
    print (blocks)
    net_info, module_list = _create_module(blocks)
    print (net_info)
    print (module_list)
      