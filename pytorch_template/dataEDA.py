# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2021/2/14 4:48 PM
=================================================='''
import sys
sys.path.append('./')
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')

import scipy.io as sio
import numpy as np
import random
import os
import cv2
import torch

NUM_FEEDBACK_BITS = 400
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2

mat = sio.loadmat('/home/users/zhenyu.yang/data/4T4R/H_4T4R.mat')



data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))



dst_dir = './data_vis_with_sort'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse
def Score(NMSE):
    score = 1 - NMSE
    return score

# data = np.transpose(data,(0,3,2,1))

# random.shuffle(data)
# data = data[:2000]
#
#

# for i in range(1000):
#
#     img = np.uint8(data[i]* 255 )
#     # img = np.concatenate([img,img*0+255],axis=-1)
#
#     img = np.concatenate([img[:,:,0],img[:,:,1]],axis=-1)
#
#     x_var = np.std(img,axis = -1)
#     var_sort = np.argsort(x_var)
#     img = img[var_sort]
#
#
#     # img_argsort = np.argsort(img,axis = -1)
#     # img_argsort = np.uint8(img_argsort/32*255)
#     # cv2.imwrite(os.path.join(dst_dir, '{}_sort.jpg'.format(str(i).zfill(8))), img_argsort)
#     #
#     #
#     # img = np.sort(img,axis = -1)
#
#     cv2.imwrite(os.path.join(dst_dir,'{}.jpg'.format(str(i).zfill(8))),img)
#
#



input = torch.FloatTensor(data).cuda()

input = input.view(len(input),24, -1)

x_var = torch.mean((input.view(len(input), 24, -1).detach() - 0.5) ** 2, dim=-1)
x_var = torch.sort(-x_var, dim=-1)[0]
x_var = torch.mean(x_var,dim = 0)

std = torch.std(input,dim = -1)
# std_mean = torch.mean(std,dim = 0)
#
# std_big = torch.where(std > 0.01,std*0+1,std*0)
# std_big_mean = torch.mean(std_big,dim = 0)
#
#
# std_big_sum = torch.sum(1 - std_big[:,:],dim = 1)
# std_big_big = torch.where(std_big_sum >= 8,std_big_sum*0+1,std_big_sum*0)
# torch.mean(std_big_big)

#
# B = len(input)
# input = input.view(B, -1)
# input = input.t()
# input = input - 0.4
#
# input_sim = torch.matmul(input, input.t())
# input_len = torch.sum(input ** 2, dim=1, keepdim=True) ** 0.5
# input_len = torch.matmul(input_len, input_len.t())
# input_sim = input_sim / input_len

# input_sim =
# std = std.t()
# data = std
#
#
# input_sim = 0
# for i in range(500):
#     temp_data = data[i::500]
#
#     input = temp_data
#     # temp_data = temp_data[:,10]
#     # input = torch.FloatTensor(temp_data).cuda()
#
#     B = len(input)
#     input = input.view(B, -1)
#     input = input.t()
#
#     temp_sim = torch.sum(torch.abs(input.unsqueeze(-1) - input.t().unsqueeze(0)), dim=1)
#
#     input_sim = input_sim + temp_sim
#
# input_sim = input_sim/len(data)
# input_sim = input_sim/input_sim.max() * 255
# input_sim = np.uint8(input_sim.cpu().numpy())
# cv2.imwrite('input_sim_std.jpg',input_sim)
# debug = 0