from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class Net(nn.Module):
    def __init__(self, n_channel, n_dim, n_split):
        super(Net, self).__init__()

        self.n_channel = n_channel
        self.n_dim = n_dim
        self.n_split = n_split
        self.n_class = 2

        self.base_net_0 = BaseNet(self.n_dim, self.n_split)
        self.base_net_1 = BaseNet(self.n_dim, self.n_split)
        self.base_net_2 = BaseNet(self.n_dim, self.n_split)
        self.base_net_3 = BaseNet(self.n_dim, self.n_split)

        ### attention
        self.out_size = 8
        self.att_channel_dim = 2
        self.W_att_channel = nn.Parameter(torch.randn(self.out_size + 1, self.att_channel_dim))
        self.v_att_channel = nn.Parameter(torch.randn(self.att_channel_dim, 1))

        ### fc
        self.fc = nn.Linear(self.out_size, self.n_class)

    def forward(self, x_0, x_1, x_2, x_3,
                k_beat_0, k_beat_1, k_beat_2, k_beat_3,
                k_rhythm_0, k_rhythm_1, k_rhythm_2, k_rhythm_3,
                k_freq):
        x_0, alpha_0, beta_0 = self.base_net_0(x_0, k_beat_0, k_rhythm_0)
        x_1, alpha_1, beta_1 = self.base_net_1(x_1, k_beat_1, k_rhythm_1)
        x_2, alpha_2, beta_2 = self.base_net_2(x_2, k_beat_2, k_rhythm_2)
        x_3, alpha_3, beta_3 = self.base_net_3(x_3, k_beat_3, k_rhythm_3)

        x = torch.stack([x_0, x_1, x_2, x_3], 1)

        # ############################################
        # ### attention on channel
        # ############################################
        k_freq = k_freq.permute(1, 0, 2)

        tmp_x = torch.cat((x, k_freq), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_channel)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        gama = torch.div(n1, n2)
        x = torch.sum(torch.mul(gama, x), 1)

        ############################################
        ### fc
        ############################################
        x = F.softmax(self.fc(x), 1)

        ############################################
        ### return
        ############################################

        att_dic = {"alpha_0": alpha_0, "beta_0": beta_0,
                   "alpha_1": alpha_1, "beta_1": beta_1,
                   "alpha_2": alpha_2, "beta_2": beta_2,
                   "alpha_3": alpha_3, "beta_3": beta_3,
                   "gama": gama}

        return x, att_dic


class BaseNet(nn.Module):
    def __init__(self, n_dim, n_split):
        super(BaseNet, self).__init__()

        self.n_dim = n_dim
        self.n_split = n_split
        self.n_seg = int(n_dim / n_split)

        ### Input: (batch size, number of channels, length of signal sequence)
        self.conv_out_channels = 64
        self.conv_kernel_size = 32
        self.conv_stride = 2
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=self.conv_out_channels,
                              kernel_size=self.conv_kernel_size,
                              stride=self.conv_stride)
        self.conv_k = nn.Conv1d(in_channels=1,
                                out_channels=1,
                                kernel_size=self.conv_kernel_size,
                                stride=self.conv_stride)
        self.att_cnn_dim = 8
        self.W_att_cnn = nn.Parameter(torch.randn(self.conv_out_channels + 1, self.att_cnn_dim))
        self.v_att_cnn = nn.Parameter(torch.randn(self.att_cnn_dim, 1))

        ### Input: (batch size, length of signal sequence, input_size)
        self.rnn_hidden_size = 32
        self.lstm = nn.LSTM(input_size=(self.conv_out_channels),
                            hidden_size=self.rnn_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.att_rnn_dim = 8
        self.W_att_rnn = nn.Parameter(torch.randn(2 * self.rnn_hidden_size + 1, self.att_rnn_dim))
        self.v_att_rnn = nn.Parameter(torch.randn(self.att_rnn_dim, 1))

        ### fc
        self.do = nn.Dropout(p=0.5)
        self.out_size = 8
        self.fc = nn.Linear(2 * self.rnn_hidden_size, self.out_size)

    def forward(self, x, k_beat, k_rhythm):
        self.batch_size = x.size()[0]

        ############################################
        ### reshape
        ############################################
        # print('orignial x:', x.size())
        x = x.view(-1, self.n_split)
        x = x.unsqueeze(1)

        k_beat = k_beat.view(-1, self.n_split)
        k_beat = k_beat.unsqueeze(1)

        ############################################
        ### conv
        ############################################
        x = F.relu(self.conv(x))

        k_beat = F.relu(self.conv_k(k_beat))

        ############################################
        ### attention conv
        ############################################
        x = x.permute(0, 2, 1)
        k_beat = k_beat.permute(0, 2, 1)
        tmp_x = torch.cat((x, k_beat), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_cnn)
        e = torch.matmul(torch.tanh(e), self.v_att_cnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)

        ############################################
        ### reshape for rnn
        ############################################
        x = x.view(self.batch_size, self.n_seg, -1)

        ############################################
        ### rnn
        ############################################

        k_rhythm = k_rhythm.unsqueeze(-1)
        o, (ht, ct) = self.lstm(x)
        tmp_o = torch.cat((o, k_rhythm), dim=-1)
        e = torch.matmul(tmp_o, self.W_att_rnn)
        e = torch.matmul(torch.tanh(e), self.v_att_rnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        beta = torch.div(n1, n2)
        x = torch.sum(torch.mul(beta, o), 1)

        ############################################
        ### fc
        ############################################
        x = F.relu(self.fc(x))
        x = self.do(x)

        return x, alpha, beta