# =======================================================================================================================
# =======================================================================================================================
import numpy as np
from modelDesign import *
import torch
import scipy.io as sio
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# =======================================================================================================================
# =======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 128  # 128
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2

# =======================================================================================================================
# =======================================================================================================================
# Data Loading
mat = sio.loadmat('/home/users/zhenyu.yang/data/4T4R/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
# data = data[-80000:]
H_test = data
test_dataset = DatasetFolder(H_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512,
                                          shuffle=False, num_workers=0,
                                          pin_memory=True)
# =======================================================================================================================
# =======================================================================================================================
# Model Loading
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS)
model_encoder = autoencoderModel.encoder


model_encoder = model_encoder.cuda()

model_encoder.load_state_dict(torch.load('..modelSubmit/encoder.pth.tar')['state_dict'])


print("weight loaded")
# =======================================================================================================================
# =======================================================================================================================
# Encoding
model_encoder.eval()
encode_feature = []
with torch.no_grad():
    for i, autoencoderInput in tqdm(enumerate(test_loader)):
        autoencoderInput = autoencoderInput.cuda()
        autoencoderOutput = model_encoder(autoencoderInput)
        autoencoderOutput = autoencoderOutput.cpu().numpy()
        if i == 0:
            encode_feature = autoencoderOutput
        else:
            encode_feature = np.concatenate(
                (encode_feature, autoencoderOutput), axis=0)

print("feedbackbits length is ", np.shape(encode_feature)[-1])
np.save('./encOutput.npy', encode_feature)
print('Finished!')
# =======================================================================================================================
# =======================================================================================================================
