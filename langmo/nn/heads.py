

import torch.nn as nn


class BaseBERTWrapper(nn.Module):
    def __init__(self, net, freeze):
        super().__init__()
        self.net = net
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

    def get_output_size(self):
        if hasattr(self.net, "pooler"):
            if hasattr(self.net.pooler, "out_features"):
                return self.net.pooler.out_features
            if hasattr(self.net.pooler, "dense"):
                return self.net.pooler.dense.out_features
        raise RuntimeError("can't estimate encoder output size")

    def forward(self, **x):
        raise NotImplementedError("Don't use this directly.")


class BertWithCLS(BaseBERTWrapper):
    def forward(self, **x):
        res = self.net(**x)["last_hidden_state"][:, 0, :]
        return res


class BertWithPooler(BaseBERTWrapper):
    def forward(self, **x):
        res = self.net(**x)["last_hidden_state"]["pooler_output"]
        return res


class BertWithLSTM(BaseBERTWrapper):
    def __init__(self, net, freeze):
        super().__init__(net, freeze)
        size_hidden = super().get_output_size()
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
        h = self.rnn(h)
        return self.lstm_out_to_tensor(h)

    def get_output_size(self):
        cnt_rnn_directions = 2
        return super().get_output_size() * cnt_rnn_directions


def get_encoder_wrapper(name):
    name = name.lower()
    wrappers = {"cls": BertWithCLS, "pooler": BertWithPooler, "lstm": BertWithLSTM}
    return wrappers[name]
