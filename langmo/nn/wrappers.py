import torch
import torch.nn as nn


class BaseBERTWrapper(nn.Module):
    def __init__(self, net, freeze):
        super().__init__()
        self.net = net
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

    @property
    def config(self):
        # TODO: add warning here
        return self.net.config

    def get_output_size(self):
        # if hasattr(self.net, "pooler"):
        #     if hasattr(self.net.pooler, "out_features"):
        #         return self.net.pooler.out_features
        #      if hasattr(self.net.pooler, "dense"):
        #          return self.net.pooler.dense.out_features
        #  raise RuntimeError("can't estimate encoder output size")
        return self.net(input_ids=torch.LongTensor([[1, 2, 3]]))["last_hidden_state"].shape[-1]

    def forward(self, **x):
        raise NotImplementedError("Don't use this directly.")


class BertWithMeanPooler(BaseBERTWrapper):
    def __init__(self, net, freeze):
        super().__init__(net, freeze)
        self.mean_pooler = nn.Linear(self.net.config.hidden_size, self.net.config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, **x):
        res = self.net(**x)["last_hidden_state"].mean(1)
        res = self.mean_pooler(res)
        return self.activation(res)


class BertWithPooler(BaseBERTWrapper):
    def forward(self, **x):
        res = self.net(**x)["pooler_output"]
        return res


class BertWithLSTM(BaseBERTWrapper):
    def __init__(self, net, freeze):
        super().__init__(net, freeze)
        size_hidden = super().get_output_size()
        self.rnn = nn.LSTM(
            input_size=size_hidden,
            hidden_size=size_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )

    def lstm_out_to_tensor(self, x):
        # TODO: check if it should be one side of one direction and another one from another
        return x[0][:, -1, :]

    def forward(self, **x):
        h = self.net(**x)["last_hidden_state"]
        h = self.rnn(h)
        return self.lstm_out_to_tensor(h)

    def get_output_size(self):
        cnt_rnn_directions = 2
        return super().get_output_size() * cnt_rnn_directions


def wrap_encoder(encoder, name, freeze):
    name = name.lower()
    wrappers = {"cls": BertWithPooler, "mean_pooler": BertWithMeanPooler, "lstm": BertWithLSTM}
    class_wrapper = wrappers[name]
    return class_wrapper(encoder, freeze)
