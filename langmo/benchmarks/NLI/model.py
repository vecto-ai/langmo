import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(self, bottom, top):
        super().__init__()
        self.bottom = bottom
        self.top = top

    def forward(self, left, right):
        self.bottom.hidden = None
        h_left = self.bottom(left)[-1]
        self.bottom.hidden = None
        h_right = self.bottom(right)[-1]
        # TODO: add [optionally] product and difference to concat
        cnc = torch.cat((h_left, h_right), 1)
        h = self.top(cnc)
        return h


class Top(nn.Module):
    def __init__(self, in_size=512):
        super().__init__()
        self.l1 = nn.Linear(in_size, 256)
        self.l2 = nn.Linear(256, 4)

    def forward(self, x):
        h = self.l1(x)
        h = F.relu(h)
        h = self.l2(h)
        return(h)


class LSTM_Encoder(nn.Module):
    def __init__(self, embs, nhid, nlayers, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embs = nn.Embedding(embs.matrix.shape[0], embs.matrix.shape[1])
        self.rnn = nn.LSTM(embs.matrix.shape[1],
                           nhid, nlayers,
                           dropout=dropout,
                           bidirectional=True)
        # self.decoder = nn.Linear(nhid, 2)
        self.init_weights(embs)
        self.hidden = None
        self.embs.weight.requires_grad = False

    def init_weights(self, embs):
        initrange = 0.1
        self.embs.weight.data = torch.from_numpy(embs.matrix)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        # print("in", input.shape)
        emb = self.drop(self.embs(input))
        # emb.unsqueeze_(0)
        # print("emb", emb.shape)
        # output, self.hidden = self.rnn(emb, self.hidden)
        output, self.hidden = self.rnn(emb, self.hidden)
        # print("rnn out", output.shape)
        #output = self.drop(output)
        #decoded = self.decoder(output[-1])
        # print("decoded", decoded)
        return output


class Net(Siamese):
    def __init__(self, embs):
        encoder = LSTM_Encoder(embs, 128, 2)
        top = Top()
        super().__init__(encoder, top)
