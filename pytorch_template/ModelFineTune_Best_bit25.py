#=======================================================================================================================
#=======================================================================================================================
import sys
sys.path.append('./')
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')
import random

# [0.0245,
# 0.0223,
# 0.0209,
# 0.0203,
# 0.0184,
# 0.0175,
# 0.0158,
# 0.0143,
# 0.0124,
# 0.0116,
# 0.0094,
# 0.0085,
# 0.0075,
# 0.0058,
# 0.0045,
# 0.0035,
# 0.0025,
# 0.0016,
# 0.0010,
# 0.0005,
# 0.0327,
# 0.0267,
# 0.0218,
# 0.0178]


# -4.5915e-03, -2.6409e-03, -1.7904e-03, -1.1895e-03, -8.8018e-04,
#         -6.6382e-04, -4.9225e-04, -3.5828e-04, -2.5724e-04, -1.8085e-04,
#         -1.2612e-04, -8.6035e-05, -5.7648e-05, -3.7803e-05, -2.4583e-05,
#         -1.5646e-05, -9.7478e-06, -5.8278e-06, -3.3062e-06, -1.7473e-06,
#         -9.2825e-07, -4.6555e-07, -2.0561e-07, -6.8867e-08


# tensor([2.7511e-03, 1.6706e-03, 9.4170e-04, 6.2568e-04, 3.9620e-04, 2.3958e-04,
#         1.3013e-04, 6.2877e-05, 2.9229e-05, 1.0500e-05, 2.6574e-06, 7.3022e-07],
#        device='cuda:0')


import numpy as np
import torch


from ModelSplitDeepSameSplitBNWithGlobal_SingleVQ_25_new import AutoEncoder,DatasetFolder,FeaLoss,NMSE,sort_input,find_max_error_line
from CRNet import WarmUpCosineAnnealingLR

import logging

import os
import torch.nn as nn
import scipy.io as sio
import os
import math
from torch.optim.lr_scheduler import CyclicLR


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting for Data
NUM_FEEDBACK_BITS = 24*6
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
# Parameters Setting for Training
BATCH_SIZE = 1024
EPOCHS = 200
# EPOCHS = 1

# BATCH_SIZE = 512

LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4



Mini_LEARNING_RATE = 1e-6

# Mini_LEARNING_RATE = 1e-5


PRINT_RREQ = 200
torch.manual_seed(1)

#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('/data/raw_data/H_4T4R.mat')

data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
# data = np.transpose(data, (0, 3, 1, 2))
split = int(data.shape[0] * 0.8)
data_train, data_test = data[:split], data[split:]
train_dataset = DatasetFolder(data_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
#=======================================================================================================================
#=======================================================================================================================
# Model Constructing
bit_list = None

bit_list = [16, 16, 11, 11, 11, 9, 9, 9, 4, 4, 2, 2]




print(EPOCHS)
print(bit_list)

# bit_list = [8 for i in range(24)]
try:
    bit_sum = sum(bit_list)
    bit_info = '_'.join([str(v) for v in bit_list])
except:
    temp_bit_list = []
    for bits in bit_list:
        temp_bit_list += bits
    bit_sum = sum(temp_bit_list)
    bit_info = '_'.join([str(v) for v in temp_bit_list])


print(bit_sum)


autoencoderModel = AutoEncoder(bit_list)
main_encoder_dir = './model_train_split'


autoencoderModel.encoder.load_state_dict(torch.load('{}/encoder.pth.tar'.format(main_encoder_dir))['state_dict'],strict=False)
autoencoderModel.decoder.load_state_dict(torch.load('{}/decoder.pth.tar'.format(main_encoder_dir))['state_dict'],strict=False)


autoencoderModel = autoencoderModel.cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=LEARNING_RATE)


fea_loss = FeaLoss()
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 1
bestNMSE = 1



optimizer = torch.optim.AdamW(autoencoderModel.parameters(), lr=0.000108821,
                              weight_decay=5.03098e-05)

scheduler = CyclicLR(optimizer=optimizer,base_lr= 1.53882e-05,max_lr=0.000195323,step_size_up= 45 * len(train_loader),cycle_momentum=False)


# optimizer = encoder_optimizer
dst_dir = './model_finetune'
print(dst_dir)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

def find_max_error_line(input,output):
    error = (input.detach() - output.detach())**2
    error = error.permute(1,0,2,3).reshape(24,-1)
    error = torch.mean(error,dim = -1).cpu().numpy()

    sort_index = np.argsort(-error)

    error = -np.sort(-error)
    error = error/np.sum(error)

    return sort_index,error


for epoch in range(EPOCHS):
    # if epoch == 50:
    #     optimizer = decoder_optimizer
    #     scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
    #                                         T_max=100 * len(train_loader),
    #                                         T_warmup=3 * len(train_loader),
    #                                         eta_min=5e-5)
    use_Quant = True

    # if  epoch <= EPOCHS / 3:
    #     use_Quant = False

    if  epoch <= 1 or epoch % 4 ==0: # just fot bit 2
        use_Quant = False



    autoencoderModel.train()


    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()

        random_data = torch.rand(*autoencoderInput.size()) * 0.2 + 0.9
        random_data = random_data.to(autoencoderInput.device)
        autoencoderInput = (autoencoderInput - 0.5)*random_data + 0.5



        autoencoderOutput,[quant_out,inner_out] = autoencoderModel(autoencoderInput,use_Quant)
        loss = fea_loss(autoencoderInput,autoencoderOutput,quant_out,inner_out,use_Quant)


        big_loss = sum(loss)
        optimizer.zero_grad()
        big_loss.backward()
        optimizer.step()
        scheduler.step()

        big_loss = big_loss.item()
        loss = ['{:.6f}'.format(v.item()) for v in loss]
        loss = ' '.join(loss)
        if i % PRINT_RREQ == 0:
            autoencoderInput = sort_input(autoencoderInput)
            sort_index,errors = find_max_error_line(autoencoderInput,autoencoderOutput)
            sort_index = ' '.join([str(v) for v in sort_index[:12]])
            errors = ' '.join(['{:.3f}'.format(v) for v in errors[:12]])

            info = 'Epoch: [{0}][{1}/{2}]\t' 'Loss {3}\t Max Error Line: {4} ; Errors : {5}'.format(epoch, i, len(train_loader), loss,sort_index,errors)
            logging.info(info)
            print(info)
            # print()
    # Model Evaluating

    autoencoderModel.eval()
    totalLoss = 0
    totalNMSE = 0
    with torch.no_grad():
        for i, autoencoderInput in enumerate(test_loader):
            autoencoderInput = autoencoderInput.cuda()

            random_data = torch.rand(*autoencoderInput.size()) * 0.1 + 0.95
            random_data = random_data.to(autoencoderInput.device)
            autoencoderInput = (autoencoderInput - 0.5) * random_data + 0.5

            autoencoderOutput, [quant_out, inner_out] = autoencoderModel(autoencoderInput, True)

            autoencoderInput = sort_input(autoencoderInput)
            totalLoss += criterion(autoencoderOutput, autoencoderInput).item() * autoencoderInput.size(0)

            totalNMSE += NMSE(autoencoderInput.detach().cpu().numpy(),autoencoderOutput.detach().cpu().numpy()) * autoencoderInput.size(0)

        averageLoss = totalLoss / len(test_dataset)
        averageNMSE = totalNMSE / len(test_dataset)

        print('averageLoss : {:6f} , averageNMSE : {:.6f}'.format(averageLoss,averageNMSE))
        if averageLoss < bestLoss or averageNMSE < bestNMSE:
            # Model saving
            # Encoder Saving
            torch.save({'state_dict': autoencoderModel.encoder.state_dict(), }, '{}/encoder.pth.tar'.format(dst_dir))
            # Decoder Saving
            torch.save({'state_dict': autoencoderModel.decoder.state_dict(), }, '{}/decoder.pth.tar'.format(dst_dir))
            print("Model saved")
            bestLoss = averageLoss
            bestNMSE = averageNMSE

#=======================================================================================================================
#=======================================================================================================================