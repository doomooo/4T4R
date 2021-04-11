#=======================================================================================================================
#=======================================================================================================================
import sys
sys.path.append('./')



import numpy as np
import torch


from ModelSplitIndex_SingleVQ import AutoEncoder,DatasetFolder,FeaLoss,NMSE,sort_input,WarmUpCosineAnnealingLR #*



import logging
import random

import os
import torch.nn as nn
import scipy.io as sio
import os


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
EPOCHS = 300
# EPOCHS = 1


LEARNING_RATE = 1e-3
LEARNING_RATE = 3e-4

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
split = int(data.shape[0] * 0.6)
data_train, data_test = data[:split], data[split:]
train_dataset = DatasetFolder(data_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
test_dataset = DatasetFolder(data_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
#=======================================================================================================================
#=======================================================================================================================
# Model Constructing
bit_list = None


bit_list = [16]

# bit_list = [14]


print(EPOCHS)

# bit_list = [8 for i in range(24)]

autoencoderModel = AutoEncoder(bit_list)
autoencoderModel = autoencoderModel.cuda()
criterion = nn.MSELoss().cuda()
optimizer = torch.optim.AdamW(autoencoderModel.parameters(), lr=LEARNING_RATE,weight_decay=2e-4)


fea_loss = FeaLoss()
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 1e8
bestNMSE = 1e8

scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=EPOCHS * len(train_loader),
                                            T_warmup=EPOCHS//20 * len(train_loader),
                                            eta_min=1e-4)

from torch.optim.lr_scheduler import CyclicLR


optimizer = torch.optim.AdamW(autoencoderModel.parameters(), lr=LEARNING_RATE,weight_decay=0.000199497)
scheduler = CyclicLR(optimizer=optimizer,base_lr=2.78506e-06,max_lr=0.00633195,step_size_up= 19 * len(train_loader),cycle_momentum=False)





bit_info = '_'.join([str(v) for v in bit_list])
dst_dir = './model_train_order'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


for epoch in range(EPOCHS):

    use_Quant = True
    autoencoderModel.train()


    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()

        # 打乱顺序，做数据增强，在最后的两个epoch不做数据增强了
        # 不过此处我不知道官方提供的数据隐含了时间和地点，如果用了这个信息的话，应该会有其他的改动

        if epoch < 298:
            B = len(autoencoderInput)
            random_data = torch.rand((len(autoencoderInput), 24))
            x_sort = torch.sort(random_data, dim=-1)[1] + torch.arange(B).unsqueeze(-1)* 24
            x_sort = x_sort.view(-1).to(autoencoderInput.device)
            autoencoderInput_temp = autoencoderInput.view(B * 24, 16, 2)
            autoencoderInput_temp = torch.index_select(autoencoderInput_temp, 0, x_sort).view(B, 24, 16, 2)
            random_data = torch.rand((len(autoencoderInput),1,1,1)).to(autoencoderInput.device)
            ratio = autoencoderInput*0 + 0.4
            autoencoderInput = torch.where(random_data < ratio,autoencoderInput_temp,autoencoderInput)



        autoencoderOutput,[quant_out,inner_out] = autoencoderModel(autoencoderInput,use_Quant)
        loss = fea_loss(autoencoderInput,autoencoderOutput,quant_out,inner_out,use_Quant)


        # if use_Quant:
        #     loss[0] = torch.abs(loss[0] - 1900) + 1900
        big_loss = sum(loss)
        optimizer.zero_grad()
        big_loss.backward()
        optimizer.step()
        scheduler.step()

        big_loss = big_loss.item()
        loss = ['{:.6f}'.format(v.item()) for v in loss]
        loss = ' '.join(loss)
        if i % PRINT_RREQ == 0:
            info = 'Epoch: [{0}][{1}/{2}]\t' 'Loss {3}\t'.format(epoch, i, len(train_loader), loss)
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

            B = len(autoencoderInput)
            random_data = torch.rand((len(autoencoderInput), 24))
            x_sort = torch.sort(random_data, dim=-1)[1] + torch.arange( B).unsqueeze(-1) * 24
            x_sort = x_sort.view(-1).to(autoencoderInput.device)
            autoencoderInput_temp = autoencoderInput.view(B * 24, 16, 2)
            autoencoderInput_temp = torch.index_select(autoencoderInput_temp,
                                                       0, x_sort).view(B, 24,
                                                                       16, 2)
            random_data = torch.rand((len(autoencoderInput), 1, 1, 1)).to(
                autoencoderInput.device)
            ratio = autoencoderInput * 0 + 0.4
            autoencoderInput = torch.where(random_data < ratio,
                                           autoencoderInput_temp,
                                           autoencoderInput)



            autoencoderOutput, [quant_out, inner_out] = autoencoderModel(autoencoderInput)

            autoencoderInput = sort_input(autoencoderInput)
            totalLoss += criterion(autoencoderOutput*100, autoencoderInput*100).item() * autoencoderInput.size(0)

        averageLoss = totalLoss / len(test_dataset)

        print('averageLoss : {:6f}'.format(averageLoss))
        if averageLoss <= bestLoss:
            # Model saving
            # Encoder Saving
            torch.save({'state_dict': autoencoderModel.encoder.state_dict(), }, '{}/encoder.pth.tar'.format(dst_dir))
            # Decoder Saving
            torch.save({'state_dict': autoencoderModel.decoder.state_dict(), }, '{}/decoder.pth.tar'.format(dst_dir))
            print("Model saved")
            bestLoss = averageLoss
#=======================================================================================================================
#=======================================================================================================================