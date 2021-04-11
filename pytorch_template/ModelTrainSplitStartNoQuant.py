#=======================================================================================================================
#=======================================================================================================================
import sys
sys.path.append('./')
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')
import random



import numpy as np
import torch

from ModelSplitDeepSameSplitOneBNWithGlobal import AutoEncoder,DatasetFolder,FeaLoss,NMSE,sort_input,find_max_error_line # very good


from CRNet import WarmUpCosineAnnealingLR
import logging

import os
import torch.nn as nn
import scipy.io as sio
import os
import math


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
EPOCHS = 400
# EPOCHS = 1


LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4

PRINT_RREQ = 100
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

bit_list = [16, 16, 11, 11,11, 9, 9, 9, 4,4, 2, 2]


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

autoencoderModel = autoencoderModel.cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(autoencoderModel.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.AdamW(autoencoderModel.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)

# decoder_optimizer = torch.optim.Adam(autoencoderModel.decoder.parameters(), lr=LEARNING_RATE)
# encoder_optimizer = torch.optim.Adam(autoencoderModel.encoder.parameters(), lr=LEARNING_RATE)



fea_loss = FeaLoss()
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 1
bestNMSE = 1

scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=EPOCHS * len(train_loader),
                                            T_warmup=EPOCHS//20 * len(train_loader),
                                            eta_min=1e-6)


# optimizer = encoder_optimizer
dst_dir = './model_train_split'.format(bit_sum,bit_info)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
print(dst_dir)
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
    if epoch <= EPOCHS / 2.5 : # epoch % 4== 0
        use_Quant = False

    autoencoderModel.train()


    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()

        random_data = torch.rand(*autoencoderInput.size())*0.2+0.9
        random_data = random_data.to(autoencoderInput.device)
        autoencoderInput = (autoencoderInput - 0.5)*random_data + 0.5




        autoencoderOutput = autoencoderModel(autoencoderInput,use_Quant)
        loss = fea_loss(autoencoderInput,autoencoderOutput)
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


            autoencoderOutput = autoencoderModel(autoencoderInput)

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