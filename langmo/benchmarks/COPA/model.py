import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, embs, nhid, nlayers, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embs = nn.Embedding(embs.matrix.shape[0], embs.matrix.shape[1])
        self.rnn = nn.LSTM(embs.matrix.shape[1], nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, 2)
        self.init_weights(embs)
        self.hidden = None
        self.embs.weight.requires_grad = False

    def init_weights(self, embs):
        initrange = 0.1
        self.embs.weight.data = torch.from_numpy(embs.matrix)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        # print("in", input.shape)
        emb = self.drop(self.embs(input))
        # emb.unsqueeze_(0)
        # print("emb", emb.shape)
        output, self.hidden = self.rnn(emb, self.hidden)
        # print("rnn out", output.shape)
        # output = self.drop(output)
        decoded = self.decoder(output[-1])
        # print("decoded", decoded)
        return decoded
