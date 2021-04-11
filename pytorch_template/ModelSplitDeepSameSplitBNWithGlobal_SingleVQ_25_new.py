#=======================================================================================================================
#=======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
from collections import Counter

NUM_FEEDBACK_BITS = 1000 #pytorch版本一定要有这个参数

Num_quan_bits = 3
KeepNum = 12
GlobalLen = 4



Encoder_Layers = [64,80,32] # 33
Decoder_Layers = [88,72,72,64]




print(Encoder_Layers)
print(Decoder_Layers)

#=======================================================================================================================
#=======================================================================================================================
# Number to Bit Defining Function Defining
class SplitLayers(nn.Module):
    def __init__(self,inchannel,outchannel,num = 4,activation = nn.ReLU()):
        super(SplitLayers, self).__init__()
        # num = num + 1

        self.num = num

        ratio = [num - i for i in range(num)]
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




def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class MinEncoder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self, feedback_bits,Channel = 2):
        super(MinEncoder, self).__init__()

        self.temp_quantize = QuantizationLayer(7)
        self.temp_dequantize = DequantizationLayer(7)

        self.channel = Channel


        activation = Mish()
        # activation = nn.LeakyReLU(negative_slope=0.3, inplace=True)

        temp_layers = []
        temp_layers.append(SplitLayers(32, Encoder_Layers[0]*feedback_bits,4,activation))
        # temp_layers.append(nn.Sigmoid())
        temp_layers.append(activation)

        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))


        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(SplitLayers(Encoder_Layers[0]*feedback_bits, Encoder_Layers[1]*feedback_bits,4,activation))
        temp_layers.append(activation)

        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(SplitLayers(Encoder_Layers[1]*feedback_bits, Encoder_Layers[2]*feedback_bits,4,activation))
        temp_layers.append(activation)

        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(nn.Linear(Encoder_Layers[2]*feedback_bits, feedback_bits*1))

        self.fc = nn.Sequential(*temp_layers)

        self.sig = nn.Sigmoid()




    def forward(self, x,quant = True):
        B = len(x)
        x = x.view(-1,self.channel,32).contiguous()
        # x = self.temp_dequantize(self.temp_quantize(x))


        out = self.fc(x)
        out = self.sig(out).view(B,-1,1)


        return out




class MinDecoder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self, feedback_bits,Channel=2):
        super(MinDecoder, self).__init__()
        self.feedback_bits = feedback_bits

        self.channel = Channel
        temp_layers = []
        activation = Mish()
        # activation = nn.Sigmoid()
        # activation = nn.LeakyReLU(negative_slope=0.3, inplace=True)


        temp_layers.append(nn.Linear(feedback_bits*1 + GlobalLen, Decoder_Layers[0]*feedback_bits))
        temp_layers.append(activation)
        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(SplitLayers(Decoder_Layers[0]*feedback_bits, Decoder_Layers[1]*feedback_bits,4,activation))
        temp_layers.append(activation)
        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(SplitLayers(Decoder_Layers[1]*feedback_bits, Decoder_Layers[2]*feedback_bits,4,activation))
        temp_layers.append(activation)


        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(SplitLayers(Decoder_Layers[2]*feedback_bits, Decoder_Layers[3]*feedback_bits,4,activation))
        temp_layers.append(activation)

        # temp_layers.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        temp_layers.append(nn.BatchNorm1d(Channel))
        temp_layers.append(nn.Linear(Decoder_Layers[3]*feedback_bits, 32))

        self.fc = nn.Sequential(*temp_layers)


        self.sig = nn.Sigmoid()

    def forward(self, x,global_info):

        out = x.view(-1, self.channel,self.feedback_bits*1).contiguous() #需使用contiguous().view(),或者可修改为reshape
        global_info = global_info.repeat(1,self.channel,1)
        out = torch.cat([out,global_info],dim = -1)

        out = self.sig(self.fc(out))
        out = out.view(-1,self.channel,16,2).contiguous() #需使用contiguous().view(),或者可修改为reshape
        return out


class Encoder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self, bit_list = None):
        super(Encoder, self).__init__()

        self.temp_quantize = QuantizationLayer(4)
        self.temp_dequantize = DequantizationLayer(4)

        self.encoder_list = []
        if bit_list is None:
            bit_list = [7 for _ in range(24)]
        self.bit_list = bit_list
        bit_count = Counter(bit_list)
        bit_count = [[k,v] for k,v in bit_count.items()]
        bit_count.sort(key=lambda x:-x[0])

        self.bit_count = bit_count

        for bit, num in self.bit_count:
            setattr(self,'encoder_{}'.format(bit),MinEncoder(bit,num))


        # 这个索引是根据训练好的模型在训练集上进行前传，统计各个压缩后数据（非0-1bit）的幅值变化情况进行排序；
        # 对于方差比较小的节点，可以认为信息量较少，可以用更少的bit进行量化；
        # 该方法进行最后做了简单尝试，并没有进一步优化，故将5%的数据从3bit变成2bit；

        std_sort = [102, 103,  88, 29, 18,25,17,96,  97,  89,  99,  90, 100,  91,  85,  83,  98,  81,
        101,  82,51,58,  79,  62,  94,  86,  92,  95,  80,  56,  54,  71,  70,  59,
         73,  76,  72,  74,   93,  87,45,  47,  43,  67,  48,  40,
         38,  77,  68,  39,  60,  11,34, 49,  50,  36, 2,  37,    35,
          32,  24,  65,  23,  27,   8,   7,  14,  13,  22,  21,   9,
         42,  12,  41,  46,  16,  61,  31,  26,  10,  20,   5,   6,  30,   1,
         15,  28,   4,  57,  52,  53,  64,   0,   3,  75,  33,  19,  55,  66,
         84,  63,  69,  44,  78]



        self.bit3_len = int(sum(bit_list) * 0.95)
        self.bit2_len = sum(bit_list) - self.bit3_len


        origin_codebook = (torch.arange(2**2).float()+0.5)/2**2
        origin_codebook = origin_codebook.view(1,1,1,-1).repeat(1,self.bit2_len,1,1)
        self.codebook_2 = nn.Parameter(origin_codebook)


        origin_codebook = (torch.arange(2**3).float()+0.5)/2**3
        origin_codebook = origin_codebook.view(1,1,1,-1).repeat(1,self.bit3_len,1,1)
        self.codebook_3 = nn.Parameter(origin_codebook)


        self.bit3_index = std_sort[-self.bit3_len:]
        self.bit2_index = std_sort[:self.bit2_len]


        # self.codebook = nn.Parameter(torch.rand(1,sum(bit_list),1,2**self.num_quan_bits))



    def forward(self, x,quant = True):
        out = []
        B,_,_,_ = x.size()

        x_var = torch.mean((x.view(B,24,-1).detach() - 0.5)**2,dim = -1)
        x_sort = torch.sort(-x_var,dim = -1)[1] + torch.arange(B).unsqueeze(-1).to(x_var.device)*24
        x_sort = x_sort.view(-1)

        x = x.view(B*24,16,2)
        x = torch.index_select(x, 0, x_sort).view(B,24,16,2)

        x = x - 0.5

        start_index = 0

        for bit,num in self.bit_count:
            end_index = start_index + num
            out.append(getattr(self,'encoder_{}'.format(bit))(x[:,start_index:end_index],quant))
            start_index = end_index

        out = torch.cat(out,dim = 1)

        out = out.view(B, -1, 1)

        out_3 = out[:,self.bit3_index]
        out_2 = out[:,self.bit2_index]



        quant3_out = VQQuant(out_3, self.codebook_3)
        quant2_out = VQQuant(out_2, self.codebook_2)

        quant_out = out*0
        quant_out[:,self.bit3_index] = quant3_out
        quant_out[:, self.bit2_index] = quant2_out


        strite_out = out + (quant_out - out).detach()
        strite_out = strite_out.view(B, -1)


        if quant:
            return strite_out.view(B,-1),quant_out,out

        return out.view(B,-1),quant_out,out



class Decoder(nn.Module):
    num_quan_bits = Num_quan_bits
    def __init__(self,  bit_list = None):
        super(Decoder, self).__init__()
        self.decoder_list = []
        if bit_list is None:
            bit_list = [7 for _ in range(24)]

        self.bit_list = bit_list
        bit_count = Counter(bit_list)
        bit_count = [[k,v] for k,v in bit_count.items()]
        bit_count.sort(key=lambda x:-x[0])

        self.bit_count = bit_count

        for bit, num in self.bit_count:
            setattr(self,'decoder_{}'.format(bit),MinDecoder(bit,num))


        temp_layers = []
        temp_layers.append(nn.Linear(sum(bit_list)*1,512))
        temp_layers.append(Mish())
        temp_layers.append(nn.BatchNorm1d(1))
        temp_layers.append(nn.Linear(512,GlobalLen))
        temp_layers.append(Mish())

        self.global_layer = nn.Sequential(*temp_layers)
        self.dequantize = DequantizationLayer(self.num_quan_bits)

    def forward(self, x,quant = True):

        #
        # if quant:
        #     x = self.dequantize(x)

        global_info = self.global_layer(x.unsqueeze(1))

        out = []
        num_quan_bits = 1

        start_index = 0
        for bit,num in self.bit_count:
            end_index = start_index + num*bit*num_quan_bits
            out.append(getattr(self,'decoder_{}'.format(bit))(x[:, start_index:end_index],global_info))
            start_index = end_index

        out = torch.cat(out,dim=1)
        last = torch.zeros_like(out[:, :1]).repeat(1,24-KeepNum,1,1) + 0.5

        # last = torch.zeros_like(out[:, :24 - KeepNum]) + 0.5
        out = torch.cat([out, last], dim=1)

        return out

class AutoEncoder(nn.Module):
    def __init__(self, bit_list=None):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(bit_list)
        self.decoder = Decoder(bit_list)
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
        self.mse = nn.MSELoss()
        self.SL1 = nn.SmoothL1Loss()
        self.nmse = NMSELoss()
        self.sim_loss = SimLoss()
        self.wing_loss = WingLoss()

    def forward(self, input, output,quant_out,inner_out,use_Quant):
        B = len(input)
        x_var = torch.mean((input.view(B,24,-1).detach() - 0.5)**2,dim = -1)
        x_sort = torch.sort(-x_var,dim = -1)[1] + torch.arange(B).unsqueeze(-1).to(x_var.device)*24
        x_sort = x_sort.view(-1)
        # x_sort = torch.sort(x_sort, dim=-1)[1]

        input = input.view(B*24,16,2)
        input = torch.index_select(input, 0, x_sort).view(B,24,16,2)

        # output = output.view(B*24,16,2)
        # output = torch.index_select(output, 0, x_sort).view(B,24,16,2).contiguous()

        # output[:,KeepNum:] = input[:,KeepNum:]

        loss = []
        loss.append(self.mse(output, input))
        # loss.append(self.wing_loss(input, output))
        loss.append(self.nmse(input, output))


        loss[0] = loss[0] * 0.6
        # loss[1] = loss[1] * 0
        loss[1] = loss[1] * 1


        if use_Quant:
            loss.append(self.mse(quant_out.detach(),inner_out)*0.25)
            loss.append(self.mse(quant_out, inner_out.detach()))
        else:
            loss.append(self.mse(quant_out.detach(),inner_out)*0.25)
            loss.append(self.mse(quant_out, inner_out.detach()))


        return loss


def sort_input(input):
    B = len(input)
    x_var = torch.mean((input.view(B, 24, -1).detach() - 0.5) ** 2,
                       dim=-1)
    x_sort = torch.sort(-x_var, dim=-1)[1] + torch.arange(B).unsqueeze(-1).to(
        x_var.device) * 24
    x_sort = x_sort.view(-1)
    # x_sort = torch.sort(x_sort, dim=-1)[1]

    input = input.view(B * 24, 16, 2)
    input = torch.index_select(input, 0, x_sort).view(B, 24, 16, 2)
    return input



def find_max_error_line(input,output):
    error = (input.detach() - output.detach())**2
    error = error.permute(1,0,2,3).reshape(24,-1)
    error = torch.mean(error,dim = -1)
    sort_index = np.argsort(-error.cpu().numpy())
    return sort_index


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




