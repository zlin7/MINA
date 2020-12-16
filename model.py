from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class AttnBeat(nn.Module):
    #Attention for the CNN step/ beat level/local information
    def __init__(self, n=3000, T=50,
                 conv_out_channels=64):
        """
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        :param conv_out_channels: also called number of filters/kernels
        TODO: We will define a network that does two things. Specifically:
            1. use one 1-D convolutional layer to capture local informatoin, on x and k_beat (see forward())
                conv: The kernel size should be set to 32, and the number of filters should be set to *conv_out_channels*. Stride should be *conv_stride*
                conv_k: same as conv, except that it has only 1 filter instead of *conv_out_channels*
            2. an attention mechanism to aggregate the convolution outputs. Specifically:
                att_W_beat: a Linear layer of shape (conv_out_channels+1, att_cnn_dim), without bias
                att_v_beat: a Linear layer of shape (att_cnn_dim, 1), without bias
        """
        super(AttnBeat, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = 32
        self.conv_stride = 2
        ### BEGIN SOLUTION
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=self.conv_out_channels,
                              kernel_size=self.conv_kernel_size,
                              stride=self.conv_stride)

        self.conv_k = nn.Conv1d(in_channels=1,
                                out_channels=1,
                                kernel_size=self.conv_kernel_size,
                                stride=self.conv_stride)
        ### END SOLUTION

        self.att_cnn_dim = 8
        ### BEGIN SOLUTION
        self.att_W_beat = nn.Linear(self.conv_out_channels + 1, self.att_cnn_dim, bias=False)
        self.att_v_beat = nn.Linear(self.att_cnn_dim, 1, bias=False)
        ### END SOLUTION
        self.init()

    def init(self):
        nn.init.normal_(self.att_W_beat.weight)
        nn.init.normal_(self.att_v_beat.weight)

    def forward(self, x, k_beat):
        """
        :param x: shape (batch, n)
        :param k_beat: shape (batch, n)
        :return:
            out: shape (batch * M, T)
            alpha: shape (batch * M, N, 1) where N is a result of convolution
        TODO:
            reshape the data - convert x/k_beat of shape (batch, n) to (batch * M, 1, T), where n = MT
            apply convolution on x and k_beat
                pass the reshaped x through self.conv, and then ReLU
                pass the reshaped k_beat through self.conv_k, and then ReLU
                concatenate the conv output of x and k_beat together
            (at this step, you might need to swap axes to align the dimensions depending on how you defined the layers)
            pass the concatenated output trough the learnable Linear transforms
                first att_W_beat, then tanh, then att_v_beat
                the output shape should be [batch*M, N=10, 1] where N is a result of conv
            to get alpha (attention values), apply softmax on the output of linear layer
                You could use F.softmax(). Be careful which dimension you apply softmax over
            aggregate the conv output of x using the attention (alpha). denote this as *out*
        """
        ### BEGIN SOLUTION
        x = x.view(-1, self.T).unsqueeze(1)
        k_beat = k_beat.view(-1, self.T).unsqueeze(1)

        x = F.relu(self.conv(x))  # Here number of filters K=64
        k_beat = F.relu(self.conv_k(k_beat))  # Conv1d(1, 1, kernel_size=(32,), stride=(2,)) => k_beat:[128*60,1,10].

        x = x.permute(0, 2, 1)  # x:[128*60,10,64]
        k_beat = k_beat.permute(0, 2, 1)
        tmp_x = torch.cat((x, k_beat), dim=-1)

        e = self.att_v_beat(torch.tanh(self.att_W_beat(tmp_x)))
        alpha = F.softmax(e, 1)
        out = torch.sum(torch.mul(alpha, x), 1)  # in the paper:o = sum_ \alpha * l
        ### END SOLUTION
        return out, alpha


class AttnRhythm(nn.Module):
    def __init__(self, n=3000, T=50, input_size=64, rhythm_out_size=8):
        """
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        :param input_size: This is the same as the # of filters/kernels in the CNN part.
        :param rhythm_out_size: output size of this netowrk
        TODO: We will define a network that does two things to handle rhythms. Specifically:
            1. use a bi-directional LSTM to process the learned local representations from the CNN part
                lstm: bidirectional, 1 layer, batch_first, and hidden_size should be set to *rnn_hidden_size*
            2. an attention mechanism to aggregate the convolution outputs. Specifically:
                att_W_rhythm: a Linear layer of shape (2 * self.rnn_hidden_size + 1, att_rnn_dim), without bias
                att_v_rhythm: a Linear layer of shape (att_rnn_dim, 1), without bias
            3. output layers
                fc: a Linear layer making the output of shape (..., self.out_size)
                do: a Dropout layer with p=0.5
        """
        #input_size is the cnn_out_channels
        super(AttnRhythm, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.input_size = input_size

        ### LSTM Input: (batch size, M, input_size)
        self.rnn_hidden_size = 32
        ### BEGIN SOLUTION
        self.lstm = nn.LSTM(input_size=self.input_size, #self.conv_out_channels,
                            hidden_size=self.rnn_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)
        ### END SOLUTION

        ### Attention mechanism
        self.att_rnn_dim = 8
        ### BEGIN SOLUTION
        self.att_W_rhythm = nn.Linear(2 * self.rnn_hidden_size + 1, self.att_rnn_dim, bias=False)
        self.att_v_rhythm = nn.Linear(self.att_rnn_dim, 1, bias=False)
        ### END SOLUTION

        ### Dropout and fully connecte layers
        self.out_size = rhythm_out_size
        ### BEGIN SOLUTION
        self.fc = nn.Linear(2 * self.rnn_hidden_size, self.out_size)
        self.do = nn.Dropout(p=0.5)
        ### END SOLUTION

        self.init()

    def init(self):
        nn.init.normal_(self.att_W_rhythm.weight)
        nn.init.normal_(self.att_v_rhythm.weight)

    def forward(self, x, k_rhythm):
        """
        :param x: shape (batch * M, self.input_size=T)
        :param k_rhythm: shape (batch, M)
        :return:
            out: shape (batch, self.out_size)
            beta: shape (batch, M, 1)
        TODO:
            reshape the data - convert x to of shape (batch, M, self.input_size), k_rhythm->(batch, M, 1)
            pass the reshaped x through lstm
            concatenate the lstm output and k_rhythm together (on the last dimension)
            pass the concatenated output trough the learnable Linear transforms
                first att_W_rhythm, then tanh, then att_v_rhythm
                the output shape should be [batch, M, 1]
            to get beta (attention values), apply softmax on the output of linear layer
            aggregate the lstm output of x using the attention (beta).
            pass the result through fully connected layer - ReLU - Dropout
            denote the final output as *out*
        """

        ### BEGIN SOLUTION
        ### reshape for rnn
        self.batch_size = int(x.size()[0] / self.M)
        x = x.view(self.batch_size, self.M, -1)
        ### rnn
        k_rhythm = k_rhythm.unsqueeze(-1)  # [128, 60, 1]
        o, (ht, ct) = self.lstm(x)  # o:[batch,60,64] (in the paper this is called h
        tmp_o = torch.cat((o, k_rhythm), dim=-1)
        e = self.att_v_rhythm(torch.tanh(self.att_W_rhythm(tmp_o)))
        beta = F.softmax(e, 1)  # [128,60,1]
        x = torch.sum(torch.mul(beta, o), 1)
        ### fc and Dropout
        x = F.relu(self.fc(x))  # [128, 64->8]
        out = self.do(x)
        ### END SOLUTION
        return out, beta

class NetFreq(nn.Module):
    def __init__(self, n_channels=4, n=3000, T=50):
        """
        :param n_channels: number of channels (F in the paper). We will need to define this many AttnBeat & AttnRhythm nets.
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        TODO: This is the main network that orchestrates the previously defined attention modules:
            1. define n_channels many AttnBeat and AttnRhythm modules. (Hint: use nn.ModuleList)
                beat_nets: for each beat_net, pass parameter conv_out_channel into the init()
                rhythm_nets: for each rhythm_net, pass conv_out_channel as input_size, and self.rhythm_out_size as the output size
            2. define frequency (channel) level attention layers
                att_W_freq: a Linear layer of shape (rhythm_out_size+1, att_channel_dim), without bias
                att_v_freq: a Linear layer of shape (att_channel_dim, 1), without bias
            3. output layer: a Linear layer for 2 classes output
        """
        super(NetFreq, self).__init__()
        self.n, self.M, self.T = n, int(n / T), T
        self.n_class = 2
        self.n_channels = n_channels
        self.conv_out_channels=64
        self.rhythm_out_size=8

        ### BEGIN SOLUTION
        self.beat_nets = nn.ModuleList()
        self.rhythm_nets = nn.ModuleList()
        for channel_i in range(self.n_channels):
            self.beat_nets.append(AttnBeat(self.n, self.T, self.conv_out_channels))
            self.rhythm_nets.append(AttnRhythm(self.n, self.T, self.conv_out_channels, self.rhythm_out_size))
        ### END SOLUTION

        ### frequency attention
        self.att_channel_dim = 2
        ### BEGIN SOLUTION
        self.att_W_freq = nn.Linear(self.rhythm_out_size + 1, self.att_channel_dim, bias=False)
        self.att_v_freq = nn.Linear(self.att_channel_dim, 1, bias=False)
        ### END SOLUTION

        ### fully-connected output layer
        ### BEGIN SOLUTION
        self.fc = nn.Linear(self.rhythm_out_size, self.n_class)
        ### END SOLUTION

        self.init()

    def init(self):
        nn.init.normal_(self.att_W_freq.weight)
        nn.init.normal_(self.att_v_freq.weight)

    def forward(self, x, k_beats, k_rhythms, k_freq):
        """
        We need to use the attention submodules to process data from each channel separately, and then pass the
            output through an attention on frequency for the final output

        :param x: shape (n_channels, batch, n)
        :param k_beats: (n_channels, batch, n)
        :param k_rhythms: (n_channels, batch, M)
        :param k_freq: (n_channels, batch, 1)
        :return:
            out: softmax output for each data point, shpae (batch, n_class)
        TODO:
            1. pass each channel of x through the corresponding beat_net, then rhythm_net.
                We will discard the attention (alpha and beta) outputs for now
            2. stack the output from 1 together into a tensor of shape (batch, n_channels, rhythm_out_size)
            3. stack the result from 2 with k_freq, and pass it through att_W_freq, tanh, att_v_freq like before
            4. apply softmax to get attention gamma, and then aggregate result from 2 using gamma
            5. pass result from 4 through the final fully connected layer, and then softmax to normalize
        """
        ### BEGIN SOLUTION
        new_x = [None for _ in range(self.n_channels)]
        att_dic = {}
        for i in range(self.n_channels):
            tx, att_dic['alpha_%d'%i] = self.beat_nets[i](x[i], k_beats[i])
            new_x[i], att_dic['beta_%d'%i] = self.rhythm_nets[i](tx, k_rhythms[i])
        x = torch.stack(new_x, 1)  # [128,8] -> [128,4,8]

        # ### attention on channel
        k_freq = k_freq.permute(1, 0, 2) #[4,128,1] -> [128,4,1]

        tmp_x = torch.cat((x, k_freq), dim=-1)
        e = self.att_v_freq(torch.tanh(self.att_W_freq(tmp_x)))
        gama = F.softmax(e, 1) #[batch, 4, 1]
        x = torch.sum(torch.mul(gama, x), 1)

        ### fc
        out = F.softmax(self.fc(x), 1)

        ### return
        att_dic['gama'] = gama
        ### END SOLUTION
        return out, gama

def test_model():
    parameter_weights = None