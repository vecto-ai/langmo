import torch
import torch.nn as nn
import torch.nn.functional as F


def get_hidden_size(model):
    if hasattr(model, "pooler"):
        if hasattr(model.pooler, "out_features"):
            return model.pooler.out_features
        if hasattr(model.pooler, "dense"):
            return (model.pooler.dense.out_features)
    raise RuntimeError("can't estimate encoder output size")


class SiameseBase(nn.Module):
    def __init__(self, bottom, top, freeze_bottom=True):
        super().__init__()
        self.bottom = bottom
        # self.bottom.requires_grad = not freeze_bottom
        self.top = top

    def combine(self, u, v):
        return torch.cat((u, v, torch.abs(u - v), u * v), 1)

    def save_pretrained(self, path):
        print("save poretrained not impolemented")


class BertWithLSTM(nn.Module):
    def __init__(self, net, freeze):
        super().__init__()
        self.net = net
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False
        size_hidden = get_hidden_size(self.net)
        self.rnn = nn.LSTM(input_size=size_hidden,
                           hidden_size=size_hidden,
                           num_layers=2,
                           batch_first=True,
                           dropout=0,
                           bidirectional=True)

    def lstm_out_to_tensor(self, x):
        return x[0][:, -1, :]

    def forward(self, **x):
        h = self.net(**x)["last_hidden_state"]
        # [:, 0, :]
        # ["pooler_output"]
        h = self.rnn(h)
        return self.lstm_out_to_tensor(h)


class SiameseLSTM(SiameseBase):
    def forward(self, left, right):
        self.bottom.hidden = None
        h_left = self.bottom(left)[-1]
        self.bottom.hidden = None
        h_right = self.bottom(right)[-1]
        cnc = self.combine(h_left, h_right)
        h = self.top(cnc)
        return h


class Siamese(SiameseBase):
    def forward(self, left, right):
        h_left = self.bottom(** left)
        h_right = self.bottom(** right)
        combined = self.combine(h_left, h_right)
        h = self.top(combined)
        return {"logits": h}


class TopMLP2(nn.Module):
    def __init__(self, in_size=512, hidden_size=768, cnt_classes=3):
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, cnt_classes)

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
        self.embs.weight.requires_grad = False

    def init_weights(self, embs):
        # initrange = 0.1
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


# TODO: call it something more self-descriptive
# class Net(Siamese):
#     def __init__(self, embs):
#         encoder = LSTM_Encoder(embs, 128, 2)
#         top = TopMLP2()
#         super().__init__(encoder, top)
