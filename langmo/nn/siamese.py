import torch
import torch.nn as nn


class SiameseBase(nn.Module):
    def __init__(self, bottom, top, freeze_bottom=True):
        super().__init__()
        self.bottom = bottom
        # self.bottom.requires_grad = not freeze_bottom
        self.top = top

    def combine(self, u, v):
        return torch.cat((u, v, torch.abs(u - v), u * v), 1)

    def save_pretrained(self, path):
        print("save poretrained not implemented")

    def forward(self, left, right):
        raise NotImplementedError("Don't use this.")


class Siamese(SiameseBase):
    def forward(self, left, right):
        h_left = self.bottom(**left)
        h_right = self.bottom(**right)
        combined = self.combine(h_left, h_right)
        h = self.top(combined)
        return {"logits": h}


# TODO: move it encoders later. but this is not head!
class LSTM_Encoder(nn.Module):
    def __init__(self, embs, nhid, nlayers, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.embs = nn.Embedding(embs.matrix.shape[0], embs.matrix.shape[1])
        self.rnn = nn.LSTM(embs.matrix.shape[1], nhid, nlayers, dropout=dropout, bidirectional=True)
        # self.decoder = nn.Linear(nhid, 2)
        self.init_weights(embs)
        self.embs.weight.requires_grad = False

    def init_weights(self, embs):
        # initrange = 0.1
        self.embs.weight.data = torch.from_numpy(embs.matrix)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # print("in", input.shape)
        emb = self.drop(self.embs(x))
        # emb.unsqueeze_(0)
        # print("emb", emb.shape)
        # output, self.hidden = self.rnn(emb, self.hidden)
        output, _hidden = self.rnn(emb)
        # print("rnn out", output.shape)
        # output = self.drop(output)
        # decoded = self.decoder(output[-1])
        # print("decoded", decoded)
        return output[:, 0, :]

# TODO: call it something more self-descriptive
# class Net(Siamese):
#     def __init__(self, embs):
#         encoder = LSTM_Encoder(embs, 128, 2)
#         top = TopMLP2()
#         super().__init__(encoder, top)
