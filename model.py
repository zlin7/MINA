from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class AttnBeat(nn.Module):
    #Attention for the CNN step/ beat level/local information
    def __init__(self, n=3000, T=50,
                 conv_out_channels=64):
        super(AttnBeat, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T

        ### Input: (batch size, number of channels, length of signal sequence)
        self.conv_out_channels = conv_out_channels
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

    def forward(self, x, k_beat):
        ##self.batch_size = x.size()[0]

        ############################################
        ### reshape
        ############################################
        x = x.view(-1, self.T).unsqueeze(1)
        k_beat = k_beat.view(-1, self.T).unsqueeze(1)
        # split length n=3000 into M=60 * K=50, x/k_beat:[128*60=7680,50]

        ############################################
        ### conv
        ############################################
        x = F.relu(self.conv(x))  # Here number of filters K=64

        k_beat = F.relu(self.conv_k(k_beat))  # Conv1d(1, 1, kernel_size=(32,), stride=(2,)) => k_beat:[128*60,1,10].
        ############################################
        ### attention conv
        ############################################
        x = x.permute(0, 2, 1)  # x:[128*60,10,64]
        k_beat = k_beat.permute(0, 2, 1)
        tmp_x = torch.cat((x, k_beat), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_cnn)
        e = torch.matmul(torch.tanh(e), self.v_att_cnn)
        alpha = F.softmax(e, 1)
        x = torch.sum(torch.mul(alpha, x), 1)  # in the paper:o = sum_ \alpha * l
        return x, alpha


class AttnRhythm(nn.Module):
    def __init__(self, n=3000, T=50, input_size=64, out_size=8):
        #input_size is the cnn_out_channels
        super(AttnRhythm, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.input_size = input_size

        ### Input: (batch size, length of signal sequence, input_size)
        self.rnn_hidden_size = 32
        self.lstm = nn.LSTM(input_size=self.input_size, #self.conv_out_channels,
                            hidden_size=self.rnn_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.att_rnn_dim = 8
        self.W_att_rnn = nn.Parameter(torch.randn(2 * self.rnn_hidden_size + 1, self.att_rnn_dim))
        self.v_att_rnn = nn.Parameter(torch.randn(self.att_rnn_dim, 1))

        ### fc
        self.do = nn.Dropout(p=0.5)
        self.out_size = out_size
        self.fc = nn.Linear(2 * self.rnn_hidden_size, self.out_size)

    def forward(self, x, k_rhythm):
        ### reshape for rnn
        self.batch_size = int(x.size()[0] / self.M)
        ############################################
        x = x.view(self.batch_size, self.M, -1)

        ############################################
        ### rnn
        ############################################
        k_rhythm = k_rhythm.unsqueeze(-1)  # [128, 60, 1]
        o, (ht, ct) = self.lstm(x)  # o:[batch,60,64] (in the paper this is called h
        tmp_o = torch.cat((o, k_rhythm), dim=-1)
        e = torch.matmul(tmp_o, self.W_att_rnn)
        e = torch.matmul(torch.tanh(e), self.v_att_rnn)
        beta = F.softmax(e, 1)  # [128,60,1]
        x = torch.sum(torch.mul(beta, o), 1)
        ############################################
        ### fc
        ############################################
        x = F.relu(self.fc(x))  # [128, 64->8]
        x = self.do(x)
        return x, beta

class NetFreq(nn.Module):
    def __init__(self,  n_channels=4, n_dim=3000, T=50):
        # T is n_split. We will split
        super(NetFreq, self).__init__()
        self.n, self.M, self.T = n_dim, int(n_dim /T), T

        self.n_class = 2
        self.n_channels = n_channels

        self.conv_out_channels=64
        self.out_size=8
        self.beat_nets = nn.ModuleList()
        self.rhythm_nets = nn.ModuleList()
        for channel_i in range(self.n_channels):
            self.beat_nets.append(AttnBeat(self.n, self.T, self.conv_out_channels))
            self.rhythm_nets.append(AttnRhythm(self.n, self.T, self.conv_out_channels, self.out_size))


        ### attention
        self.att_channel_dim = 2
        self.W_att_channel = nn.Parameter(torch.randn(self.out_size + 1, self.att_channel_dim))
        self.v_att_channel = nn.Parameter(torch.randn(self.att_channel_dim, 1))

        ### fc
        self.fc = nn.Linear(self.out_size, self.n_class)

    def forward(self, x_0, x_1, x_2, x_3,
                k_beat_0, k_beat_1, k_beat_2, k_beat_3,
                k_rhythm_0, k_rhythm_1, k_rhythm_2, k_rhythm_3,
                k_freq):
        return self.forward_([x_0, x_1, x_2, x_3],
                             [k_beat_0, k_beat_1, k_beat_2, k_beat_3],
                             [k_rhythm_0, k_rhythm_1, k_rhythm_2, k_rhythm_3],
                             k_freq)

    def forward_(self, x, k_beats, k_rhythms, k_freq):
        #x, k_beats, k_rhythms, k_freq = x.permute(1,0,2), k_beats.permute(1,0,2), k_rhythms.permute(1,0,2), k_freq.permute(1,0,2)

        new_x = [None for _ in range(self.n_channels)]
        att_dic = {}
        for i in range(self.n_channels):
            tx, att_dic['alpha_%d'%i] = self.beat_nets[i](x[i], k_beats[i])
            new_x[i], att_dic['beta_%d'%i] = self.rhythm_nets[i](tx, k_rhythms[i])
        x = torch.stack(new_x, 1)  # [128,8] -> [128,4,8]

        # ############################################
        # ### attention on channel
        # ############################################
        k_freq = k_freq.permute(1, 0, 2) #[4,128,1] -> [128,4,1]

        tmp_x = torch.cat((x, k_freq), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_channel)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        gama = F.softmax(e, 1) #[batch, 4, 1]
        #ipdb.set_trace()
        x = torch.sum(torch.mul(gama, x), 1)

        ############################################
        ### fc
        ############################################
        x = F.softmax(self.fc(x), 1)

        ############################################
        ### return
        ############################################
        att_dic['gama'] = gama
        return x, att_dic
