#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

NUM_FEEDBACK_BITS = 1000 #pytorch版本一定要有这个参数


Num_quan_bits = 3
BitNum = 16



Encoder_Layers = [300,600,600] # 21
Decoder_Layers = [800,600,400]

print(Encoder_Layers)
print(Decoder_Layers)

#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining

class SplitLayers(nn.Module):
    def __init__(self,inchannel,outchannel,num = 4,activation = nn.ReLU()):
        super(SplitLayers, self).__init__()

        num = num -1

        self.num = num

        ratio = [(num - i)**1 for i in range(num)]
        ratio = [v/sum(ratio) for v in ratio]


        out_list = [int(outchannel*v) for v in ratio]
        out_list[-1] = outchannel - sum(out_list[:-1])

        self.out_list = out_list


        standart_len = max(inchannel,outchannel)

        last_layer = inchannel
        for i in range(num):
            temp_in = last_layer
            temp_out = out_list[i] + int(standart_len*sum(ratio[i+1:]))
            last_layer = int(standart_len*sum(ratio[i+1:]))

            layer = nn.Sequential(nn.Linear(temp_in,temp_out),activation)

            setattr(self, 'layer_{}'.format(i), layer)


    def forward(self, x):
        out = []
        for i in range(self.num):
            x = getattr(self,'layer_{}'.format(i))(x)
            out.append(x[...,:self.out_list[i]])
            x = x[...,self.out_list[i]:]
        out = torch.cat(out,dim = -1)
        return out


from torch.optim.lr_scheduler import _LRScheduler
class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]




class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        #b, c = grad_output.shape
        #grad_bit = grad_output.repeat(1, 1, ctx.constant)
        #return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def VQQuant(x,codebook):
    x = x.unsqueeze(-1)

    distance = (x - codebook)**2
    distance = distance.detach()
    distance = torch.sum(distance,dim = 2,keepdims = True)

    min_dis = torch.min(distance,dim = -1,keepdims = True)[0]

    min = codebook*0  + torch.tensor(float('-inf'))
    out = torch.where(distance == min_dis,codebook,min )
    out = torch.max(out,dim = -1)[0]

    return out



class MinEncoderOrder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self, feedback_bits):
        super(MinEncoderOrder, self).__init__()


        temp_layers = []
        temp_layers.append(SplitLayers(24, Encoder_Layers[0]))
        temp_layers.append(Mish())



        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(SplitLayers(Encoder_Layers[0], Encoder_Layers[1]))
        temp_layers.append(Mish())


        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(SplitLayers(Encoder_Layers[1], Encoder_Layers[2]))
        temp_layers.append(Mish())


        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(nn.Linear(Encoder_Layers[2], feedback_bits*1))

        self.fc = nn.Sequential(*temp_layers)

        self.sig = nn.Sigmoid()



        self.codebook = nn.Parameter(torch.rand(1,feedback_bits,1,2**self.num_quan_bits))


    def forward(self, x,quant = True):
        B = len(x)
        x = x.view(-1,1,24).contiguous()

        out = self.fc(x)
        out = self.sig(out).view(B,-1,1)



        quant_out = VQQuant(out, self.codebook)
        strite_out = out + (quant_out - out).detach()
        strite_out = strite_out.view(B, -1)


        if quant:
            return strite_out.view(B,-1),quant_out,out

        return out.view(B,-1),quant_out,out


class MinDecoderOrder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self, feedback_bits):
        super(MinDecoderOrder, self).__init__()
        self.feedback_bits = feedback_bits

        temp_layers = []
        temp_layers.append(SplitLayers(feedback_bits*1, Decoder_Layers[0]))
        temp_layers.append(Mish())
        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(SplitLayers(Decoder_Layers[0], Decoder_Layers[1]))
        temp_layers.append(Mish())
        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(SplitLayers(Decoder_Layers[1], Decoder_Layers[2]))

        temp_layers.append(Mish())
        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(nn.Linear(Decoder_Layers[2], 24))

        self.fc = nn.Sequential(*temp_layers)


        # self.sig = nn.Sigmoid()

    def forward(self, x,quant = True):
        # if quant:
        #     x = self.dequantize(x)

        out = x.view(-1, 1,self.feedback_bits*1).contiguous() #需使用contiguous().view(),或者可修改为reshape
        # out = self.sig(self.fc(out))
        out = self.fc(out)

        out = out.view(-1,24).contiguous() #需使用contiguous().view(),或者可修改为reshape
        return out


class EncoderOrder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self, bit_list = None):
        super(EncoderOrder, self).__init__()


        self.encoder = MinEncoderOrder(BitNum)


    def forward(self, x,quant = True):
        out = []
        B,_,_,_ = x.size()

        x = torch.mean((x.view(B,24,-1).detach() - 0.5)**2,dim = -1)
        x = torch.sort(-x, dim=-1)[1].float()
        x = torch.sort(x, dim=-1)[1].float()
        x = torch.where(x > 18, (x - 18) * 0.2 + 18, x)

        strite_out, quant_out, out = self.encoder(x,quant)

        return strite_out,quant_out,out


class DecoderOrder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self,  bit_list = None):
        super(DecoderOrder, self).__init__()
        self.decoder = MinDecoderOrder(BitNum)

    def forward(self, x,quant = True):
        out = self.decoder(x,quant)
        return out



class AutoEncoder(nn.Module):
    def __init__(self, bit_list=None):
        super(AutoEncoder, self).__init__()
        self.encoder = EncoderOrder(bit_list)
        self.decoder = DecoderOrder(bit_list)
    def forward(self, x,quant = True):
        strite_out, quant_out, inner_out = self.encoder(x,quant)
        out = self.decoder(strite_out,quant)
        return out,[quant_out,inner_out]


#=======================================================================================================================
#=======================================================================================================================
# NMSE Function Defining
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


def NMSE_cuda(x, x_hat):
    x_real = x[:, :, :, 0].view(len(x), -1) - 0.5
    x_imag = x[:, :, :, 1].view(len(x), -1) - 0.5
    x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real ** 2 + x_imag ** 2, axis=1)
    mse = torch.sum((x_real - x_hat_real) ** 2 + (x_imag - x_hat_imag) ** 2,
                    axis=1)
    nmse = mse / power
    return nmse


class NMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction
        self.loss_history = []
        self.max_len = 100
        self.iter = -1

    def forward(self, x, x_hat):
        nmse = NMSE_cuda(x, x_hat)
        # self.loss_history.append(nmse.detach())
        # self.iter += 1
        # if len(self.loss_history) > 100:
        #     _ = self.loss_history.pop(0)
        # if self.iter % 200 == 0:
        #     loss_history = torch.stack(self.loss_history,dim = -1)
        #     loss_history = torch.mean(loss_history, dim=-1).cpu().numpy()
        #     print(loss_history)

        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
        else:
            nmse = torch.sum(nmse)
        return nmse



class SimLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()


    def calSim(self,input):

        B = len(input)
        input = input.view(B,-1)
        input_sim = torch.matmul(input, input.t())
        input_len = torch.sum(input**2,dim=1,keepdim = True)**0.5
        input_len = torch.matmul(input_len, input_len.t())
        input_sim = input_sim/input_len

        return input_sim


    def forward(self, input, output):
        input = input - 0.5
        output = output - 0.5

        input_sim = self.calSim(input)
        output_sim = self.calSim(output)

        loss = self.mse(input_sim*5,output_sim*5)

        loss_d = loss.detach()
        zero = loss_d*0

        loss = torch.where(loss_d > 0.05,zero+1,zero)*loss
        return torch.mean(loss)

class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, target,pred):
        times = 80
        y = target*times
        y_hat = pred*times
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C

        loss = (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
        loss = loss/times
        return loss



class FeaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'none')
        self.SL1 = nn.SmoothL1Loss()
        self.nmse = NMSELoss()
        self.sim_loss = SimLoss()
        self.wing_loss = WingLoss()

        self.mean_mse = nn.MSELoss()

    def forward(self, input, output,quant_out,inner_out,use_Quant):
        B = len(input)
        input = torch.mean((input.view(B,24,-1).detach() - 0.5)**2,dim = -1)
        input = torch.sort(-input, dim=-1)[1].float()
        input = torch.sort(input, dim=-1)[1].float()
        input = torch.where(input > 18, (input - 18) * 0.2 + 18, input)

        loss = []

        weight = (24 - input)
        mse = self.mse(output*100, input*100) *  weight
        mse = torch.mean(mse)

        loss.append(mse)

        # 该做法参考自VQ-VAE
        if use_Quant:
            loss.append(self.mean_mse(quant_out.detach(),inner_out)*0.25)
            loss.append(self.mean_mse(quant_out, inner_out.detach()))
        else:
            loss.append(self.mean_mse(quant_out, inner_out.detach()))

        return loss


def sort_input(input):
    B = len(input)
    input = torch.mean((input.view(B, 24, -1).detach() - 0.5) ** 2, dim=-1)
    input = torch.sort(-input, dim=-1)[1].float()
    input = torch.sort(input, dim=-1)[1].float()
    input = torch.where(input > 18, (input - 18) * 0.2 + 18, input)

    # 上述大于18做截断与缩小的原因在于18以上就是幅值很小的了，他们的顺序对结果影响不大，模型不需要过分的学习他们的排序，只要知道个大概就可以了

    return input





#=======================================================================================================================
#=======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData
    def __getitem__(self, index):
        return self.matdata[index]
    def __len__(self):
        return self.matdata.shape[0]




