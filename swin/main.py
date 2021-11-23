# -*- coding: utf-8 -*-
# @Author  : younger
# @Date    : 2021/10/26 14:54
# Software : PyCharm
# version： Python 3.7
# @File    : main.py

import gzip, struct
import numpy as np
from ipdb import  set_trace
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader
from dataload import deal
import math
# import copy
#model.py
from swin_transformer import SwinTransformer
import torch.nn as nn
import torch


batch = 4
epoch = 1  
root = '/media/yr/新加卷1/ly/SAR/00Original/'   #图片的根目录
train_loader = deal(root, batch=batch,train_rate = 0.8)['train']
model = SwinTransformer().cuda()
print("swin_T start")
for batch_idx, (data, target) in enumerate(train_loader):
    model(data.cuda())
    break