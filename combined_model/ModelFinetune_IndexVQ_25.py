#=======================================================================================================================
#=======================================================================================================================
import sys
sys.path.append('./')
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.insert(0,'/home/users/zhenyu.yang/data/env/transformer_related/')




import numpy as np
import torch

from ModelSplitSameSplitOneBNWithGroup_IndexVQ_single_25 import AutoEncoder,DatasetFolder,FeaLoss,NMSE,sort_input,find_max_error_line #*



import logging

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
# BATCH_SIZE = 512

EPOCHS = 100

LEARNING_RATE = 2e-5
# LEARNING_RATE = 1e-5


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

bit_list = [1] # 没有实质用处


main_encoder_dir = '../pytorch_template/model_finetune'


order_encoder_dir = '../train_order/model_train_order'



autoencoderModel = AutoEncoder(bit_list)
autoencoderModel = AutoEncoder(bit_list)
autoencoderModel.encoder.order_encoder.load_state_dict(torch.load('{}/encoder.pth.tar'.format(order_encoder_dir))['state_dict'])
autoencoderModel.encoder.order_decoder.load_state_dict(torch.load('{}/decoder.pth.tar'.format(order_encoder_dir))['state_dict'],strict=False)
autoencoderModel.decoder.order_decoder.load_state_dict(torch.load('{}/decoder.pth.tar'.format(order_encoder_dir))['state_dict'],strict=False)

autoencoderModel.encoder.order_decoder.decoder.codebook = autoencoderModel.encoder.order_encoder.encoder.codebook
autoencoderModel.decoder.order_decoder.decoder.codebook = autoencoderModel.encoder.order_encoder.encoder.codebook


autoencoderModel.encoder.load_state_dict(torch.load('{}/encoder.pth.tar'.format(main_encoder_dir))['state_dict'],strict=False)
autoencoderModel.decoder.load_state_dict(torch.load('{}/decoder.pth.tar'.format(main_encoder_dir))['state_dict'],strict=False)

autoencoderModel.encoder.codebook_2 = autoencoderModel.encoder.codebook_2
autoencoderModel.decoder.codebook_3 = autoencoderModel.encoder.codebook_3



autoencoderModel = autoencoderModel.cuda()
criterion = nn.MSELoss().cuda()


autoencoderModel.encoder.order_encoder.requires_grad = False
autoencoderModel.encoder.order_decoder.requires_grad = False
autoencoderModel.decoder.order_decoder.requires_grad = False


autoencoderModel.encoder.codebook_2.requires_grad = False
autoencoderModel.encoder.codebook_3.requires_grad = False

autoencoderModel.decoder.codebook_2.requires_grad = False
autoencoderModel.decoder.codebook_3.requires_grad = False

# optimizer = torch.optim.AdamW(autoencoderModel.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)

# decoder_optimizer = torch.optim.Adam(autoencoderModel.decoder.parameters(), lr=LEARNING_RATE)
# encoder_optimizer = torch.optim.Adam(autoencoderModel.encoder.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, autoencoderModel.parameters()), lr=LEARNING_RATE)


fea_loss = FeaLoss()
#=======================================================================================================================
#=======================================================================================================================
# Model Training and Saving
bestLoss = 1
bestNMSE = 1

# scheduler  = WarmUpCosineAnnealingLR(optimizer=optimizer,
#                                             T_max=EPOCHS * len(train_loader),
#                                             T_warmup=EPOCHS//20 * len(train_loader),
#                                             eta_min=1e-5)


# optimizer = encoder_optimizer
bit_info = '_'.join([str(v) for v in bit_list])
dst_dir = './model_finetune'

print(dst_dir)
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


for epoch in range(EPOCHS):

    use_Quant = True
    if epoch % 4 == 0:
        use_Quant = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, autoencoderModel.parameters()),
            lr=LEARNING_RATE*0.3)
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, autoencoderModel.parameters()),
            lr=LEARNING_RATE)


    autoencoderModel.train()


    for i, autoencoderInput in enumerate(train_loader):
        autoencoderInput = autoencoderInput.cuda()

        autoencoderOutput = autoencoderModel(autoencoderInput,use_Quant)
        if use_Quant:
            loss = fea_loss(autoencoderInput,autoencoderOutput)
        else:
            autoencoderOutput,[quant_out,out] = autoencoderOutput
            loss = fea_loss(autoencoderInput,autoencoderOutput,quant_out,out,use_Quant=False)


        big_loss = sum(loss)
        optimizer.zero_grad()
        big_loss.backward()
        optimizer.step()
        # scheduler.step()

        big_loss = big_loss.item()
        loss = ['{:.6f}'.format(v.item()) for v in loss]
        loss = ' '.join(loss)
        if i % PRINT_RREQ == 0:
            autoencoderInput = sort_input(autoencoderInput)
            sort_index = find_max_error_line(autoencoderInput,autoencoderOutput)
            sort_index = ' '.join([str(v) for v in sort_index[:5]])
            info = 'Epoch: [{0}][{1}/{2}]\t' 'Loss {3}\t Max Error Line: {4}'.format(epoch, i, len(train_loader), loss,sort_index)
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
            autoencoderOutput = autoencoderModel(autoencoderInput)

            # autoencoderInput = sort_input(autoencoderInput)
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