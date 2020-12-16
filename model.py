from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class KnowledgeAttn(nn.Module):
    def __init__(self, input_features, attn_dim):
        """
        This is the general knowledge-guided attention module.
        It will transform the input and knowledge with 2 linear layers, computes attention, and then aggregate.
        :param input_features: the number of features for each
        :param attn_dim: the number of hidden nodes in the attention mechanism
        TODO:
            define the following 2 linear layers WITHOUT bias (with the names provided)
                att_W: a Linear layer of shape (input_features + n_knowledge, attn_dim)
                att_v: a Linear layer of shape (attn_dim, 1)
            init the weights using self.init() (already given)
        """
        super(KnowledgeAttn, self).__init__()
        self.input_features = input_features
        self.attn_dim = attn_dim
        self.n_knowledge = 1

        ### BEGIN SOLUTION
        self.att_W = nn.Linear(self.input_features + self.n_knowledge, self.attn_dim, bias=False)
        self.att_v = nn.Linear(self.attn_dim, 1, bias=False)
        ### END SOLUTION

        self.init()

    def init(self):
        nn.init.normal_(self.att_W.weight)
        nn.init.normal_(self.att_v.weight)

    @classmethod
    def attention_sum(cls, x, attn):
        """

        :param x: of shape (-1, D, nfeatures)
        :param attn: of shape (-1, D, 1)
        TODO: return the weighted sum of x along the middle axis with weights even in attn. output shoule be (-1, nfeatures)
        """
        ### BEGIN SOLUTION
        return torch.sum(torch.mul(attn, x), 1)
        ### END SOLUTION


    def forward(self, x, k):
        """
        :param x: shape of (-1, D, input_features)
        :param k: shape of (-1, D, 1)
        :return:
            out: shape of (-1, input_features), the aggregated x
            attn: shape of (-1, D, 1)
        TODO:
            concatenate the input x and knowledge k together (on the last dimension)
            pass the concatenated output through the learnable Linear transforms
                first att_W, then tanh, then att_v
                the output shape should be (-1, D, 1)
            to get attention values, apply softmax on the output of linear layer
                You could use F.softmax(). Be careful which dimension you apply softmax over
            aggregate x using the attention values via self.attention_sum, and return
        """
        ### BEGIN SOLUTION
        tmp = torch.cat([x, k], dim=-1)
        e = self.att_v(torch.tanh(self.att_W(tmp)))
        attn = F.softmax(e, 1)
        out = self.attention_sum(x, attn)
        ### END SOLUTION
        return out, attn


def float_tensor_equal(a, b, eps=1e-3):
    return torch.norm(a-b).abs().max().tolist() < eps

def testKnowledgeAttn():
    m = KnowledgeAttn(2, 2)
    m.att_W.weight.data = torch.tensor([[0.3298,  0.7045, -0.1067],
                                        [0.9656,  0.3090,  1.2627]], requires_grad=True)
    m.att_v.weight.data = torch.tensor([[-0.2368,  0.5824]], requires_grad=True)

    x = torch.tensor([[[-0.6898, -0.9098], [0.0230,  0.2879], [-0.2534, -0.3190]],
                      [[ 0.5412, -0.3434], [0.0289, -0.2837], [-0.4120, -0.7858]]])
    k = torch.tensor([[ 0.5469,  0.3948, -1.1430], [0.7815, -1.4787, -0.2929]]).unsqueeze(2)
    out, attn = m(x, k)

    tout = torch.tensor([[-0.2817, -0.2531], [0.2144, -0.4387]])
    tattn = torch.tensor([[[0.3482], [0.4475], [0.2043]],
                          [[0.5696], [0.1894], [0.2410]]])
    assert float_tensor_equal(attn, tattn), "The attention values are wrong"
    assert float_tensor_equal(out, tout), "output of the attention module is wrong"


#============================================================

class BeatNet(nn.Module):
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
                attn: KnowledgeAttn with input_features equaling conv_out_channels, and attn_dim=att_cnn_dim
        """
        super(BeatNet, self).__init__()
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
        self.attn = KnowledgeAttn(self.conv_out_channels, self.att_cnn_dim)
        ### END SOLUTION

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
            (at this step, you might need to swap axes to align the dimensions depending on how you defined the layers)
            pass the conv'd x and conv'd knowledge through attn to get the output (*out*) and alpha
        """
        ### BEGIN SOLUTION
        x = x.view(-1, self.T).unsqueeze(1)
        k_beat = k_beat.view(-1, self.T).unsqueeze(1)

        x = F.relu(self.conv(x))  # Here number of filters K=64
        k_beat = F.relu(self.conv_k(k_beat))  # Conv1d(1, 1, kernel_size=(32,), stride=(2,)) => k_beat:[128*60,1,10].

        x = x.permute(0, 2, 1)  # x:[128*60,10,64]
        k_beat = k_beat.permute(0, 2, 1)
        out, alpha = self.attn(x, k_beat)
        ### END SOLUTION
        return out, alpha


class RhythmNet(nn.Module):
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
                attn: KnowledgeAttn with input_features equaling lstm output, and attn_dim=att_rnn_dim
            3. output layers
                fc: a Linear layer making the output of shape (..., self.out_size)
                do: a Dropout layer with p=0.5
        """
        #input_size is the cnn_out_channels
        super(RhythmNet, self).__init__()
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
        self.attn = KnowledgeAttn(2 * self.rnn_hidden_size, self.att_rnn_dim)
        ### END SOLUTION

        ### Dropout and fully connecte layers
        self.out_size = rhythm_out_size
        ### BEGIN SOLUTION
        self.fc = nn.Linear(2 * self.rnn_hidden_size, self.out_size)
        self.do = nn.Dropout(p=0.5)
        ### END SOLUTION



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
            pass the lstm output and knowledge through attn
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

        x, beta = self.attn(o, k_rhythm)
        ### fc and Dropout
        x = F.relu(self.fc(x))  # [128, 64->8]
        out = self.do(x)
        ### END SOLUTION
        return out, beta

class FreqNet(nn.Module):
    def __init__(self, n_channels=4, n=3000, T=50):
        """
        :param n_channels: number of channels (F in the paper). We will need to define this many BeatNet & RhythmNet nets.
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        TODO: This is the main network that orchestrates the previously defined attention modules:
            1. define n_channels many BeatNet and RhythmNet modules. (Hint: use nn.ModuleList)
                beat_nets: for each beat_net, pass parameter conv_out_channel into the init()
                rhythm_nets: for each rhythm_net, pass conv_out_channel as input_size, and self.rhythm_out_size as the output size
            2. define frequency (channel) level knowledge-guided attention module
                attn: KnowledgeAttn with input_features equaling rhythm_out_size, and attn_dim=att_channel_dim
            3. output layer: a Linear layer for 2 classes output
        """
        super(FreqNet, self).__init__()
        self.n, self.M, self.T = n, int(n / T), T
        self.n_class = 2
        self.n_channels = n_channels
        self.conv_out_channels=64
        self.rhythm_out_size=8

        ### BEGIN SOLUTION
        self.beat_nets = nn.ModuleList()
        self.rhythm_nets = nn.ModuleList()
        for channel_i in range(self.n_channels):
            self.beat_nets.append(BeatNet(self.n, self.T, self.conv_out_channels))
            self.rhythm_nets.append(RhythmNet(self.n, self.T, self.conv_out_channels, self.rhythm_out_size))
        ### END SOLUTION

        ### frequency attention
        self.att_channel_dim = 2
        ### BEGIN SOLUTION
        self.attn = KnowledgeAttn(self.rhythm_out_size, self.att_channel_dim)
        ### END SOLUTION

        ### fully-connected output layer
        ### BEGIN SOLUTION
        self.fc = nn.Linear(self.rhythm_out_size, self.n_class)
        ### END SOLUTION


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
            3. pass result from 2 and k_freq through attention module, to get the aggregated result and gama
            4. pass aggregated result from 3 through the final fully connected layer.
            5. Apply Softmax to normalize output to a probability distribution (over 2 classes)
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
        x, gama = self.attn(x, k_freq)

        ### fc
        out = F.softmax(self.fc(x), 1)

        ### return
        att_dic['gama'] = gama
        ### END SOLUTION
        return out, gama




if __name__ == '__main__':
    testKnowledgeAttn()
