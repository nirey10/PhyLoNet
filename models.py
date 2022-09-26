import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.init_conv = nn.Conv2d(in_channels=input_dim * 2, out_channels=input_dim, kernel_size=self.kernel_size,
                                   stride=(1, 1),
                                   padding=self.padding)
        self.F = nn.Sequential()
        self.F.add_module('bn1', nn.GroupNorm(4, input_dim))
        self.F.add_module('conv1',
                          nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size,
                                    stride=(1, 1), padding=self.padding))
        # self.F.add_module('f_act1', nn.LeakyReLU(negative_slope=0.1))
        self.F.add_module('conv2',
                          nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1, 1), stride=(1, 1),
                                    padding=(0, 0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                                  out_channels=self.input_dim,
                                  kernel_size=(3, 3),
                                  padding=(1, 1), bias=self.bias)

    def forward(self, x, hidden):  # x [batch_size, hidden_dim, height, width]
        # x = self.init_conv(x)

        hidden_tilde = hidden + self.F(hidden)  # prediction

        combined = torch.cat([x, hidden_tilde], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)

        next_hidden = hidden_tilde + K * (x - hidden_tilde)  # correction , Haddamard product
        return next_hidden


class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            #    cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]

            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j - 1], self.H[j])

        return self.H, self.H

    def initHidden(self, batch_size):
        self.H = []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, H):
        self.H = H


class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()

        self.height, self.width = input_shape
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)

    # we implement LSTM that process only one timestep
    def forward(self, x, hidden):  # x [batch, hidden_dim, width, height]
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size, device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [], []
        self.device = device

        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i - 1]
            print('layer ', i, 'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                           input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dims[i],
                                           kernel_size=self.kernel_size))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False):  # input_ [batch_size, 1, channels, width, height]
        batch_size = input_.data.size()[0]
        if (first_timestep):
            self.initHidden(batch_size)  # init Hidden at each forward start

        for j, cell in enumerate(self.cell_list):
            if j == 0:  # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j], self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j - 1], (self.H[j], self.C[j]))

        return (self.H, self.C), self.H  # (hidden, output)

    def initHidden(self, batch_size):
        self.H, self.C = [], []
        for i in range(self.n_layers):
            self.H.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))
            self.C.append(
                torch.zeros(batch_size, self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device))

    def setHidden(self, hidden):
        H, C = hidden
        self.H, self.C = H, C


class EncoderRNN(torch.nn.Module):
    def __init__(self, phycell, convlstm, device, image_size):
        super(EncoderRNN, self).__init__()
        self.image_cnn_enc = Encoder_cycle_noRes(im_size=image_size).to(device)  # image encoder 64x64x1 -> 16x16x64
        self.image_cnn_dec = Decoder_cycle(input_nc=image_size).to(device)  # image decoder 16x16x64 -> 64x64x1
        self.image_size = image_size
        self.phycell = phycell.to(device)
        self.convlstm = convlstm.to(device)

    def forward(self, input, first_timestep=False, decoding=False):

        if decoding:  # input=None in decoding phase
            output_phys = None
        else:
            output_phys = self.image_cnn_enc(input)
        output_conv = self.image_cnn_enc(input)

        hidden1, output1 = self.phycell(output_phys, first_timestep)
        hidden2, output2 = self.convlstm(output_conv, first_timestep)

        out_phys = torch.sigmoid(self.image_cnn_dec(output1[-1]))  # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.image_cnn_dec(output2[-1]))

        concat = output1[-1] + output2[-1]

        output_image = torch.sigmoid(self.image_cnn_dec(concat))
        return out_phys, hidden1, output_image, out_phys, out_conv

class Encoder_cycle_noRes(nn.Module):
    def __init__(self, im_size, input_nc=3):
        super(Encoder_cycle_noRes, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 16, 7),
                    nn.InstanceNorm2d(16),
                    nn.ReLU(inplace=True) ]

        iterations = int(math.log2(im_size) - 4)
        # Downsampling
        in_features = 16
        out_features = in_features*2
        for _ in range(iterations):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Decoder_cycle(nn.Module):
    def __init__(self, input_nc=64, output_nc=3):
        super(Decoder_cycle, self).__init__()

        in_features = input_nc
        model = []

        iterations = int(math.log2(input_nc) - 4)
        out_features = in_features // 2
        for _ in range(iterations):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(16, output_nc, 7)]  # removed tanh activation

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

