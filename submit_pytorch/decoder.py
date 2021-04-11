#=======================================================================================================================
#=======================================================================================================================
import numpy as np
from modelDesign import *
import torch
import scipy.io as sio
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 100 #128
CHANNEL_SHAPE_DIM1 = 24
CHANNEL_SHAPE_DIM2 = 16
CHANNEL_SHAPE_DIM3 = 2
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
mat = sio.loadmat('/home/users/zhenyu.yang/data/4T4R/H_4T4R.mat')
data = mat['H_4T4R']
data = data.astype('float32')
data = np.reshape(data, (len(data), CHANNEL_SHAPE_DIM1, CHANNEL_SHAPE_DIM2, CHANNEL_SHAPE_DIM3))
# data = data[-80000:]
H_test = data
# encOutput Loading
encode_feature = np.load('./encOutput.npy')
test_dataset = DatasetFolder(encode_feature)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

#=======================================================================================================================
#=======================================================================================================================
# Model Loading and Decoding
autoencoderModel = AutoEncoder(NUM_FEEDBACK_BITS)
model_decoder = autoencoderModel.decoder


model_decoder = model_decoder.cuda()
model_decoder.load_state_dict(torch.load('./modelSubmit/decoder.pth.tar')['state_dict'])


print("weight loaded")
model_decoder.eval()
H_pre = []
with torch.no_grad():
    for i, decoderOutput in tqdm(enumerate(test_loader)):
        # convert numpy to Tensor
        decoderOutput = decoderOutput.cuda()
        output = model_decoder(decoderOutput)
        output = output.cpu().numpy()
        if i == 0:
            H_pre = output
        else:
            H_pre = np.concatenate((H_pre, output), axis=0)

print(NMSE(H_test, H_pre) )
if (NMSE(H_test, H_pre) < 0.1):
    print('Valid Submission')
    print('The Score is ' + np.str(1.0 - NMSE(H_test, H_pre)))
print('Finished!')
#=======================================================================================================================
#=======================================================================================================================










