import torch
import torch.nn as nn
SIZE_EMB = 256


class Net(nn.Module):
    def __init__(self, size_vocab):
        super().__init__()
        self.embed = nn.Embedding(size_vocab, SIZE_EMB)
        self.lstm_1 = nn.LSTM(SIZE_EMB, 512)
        self.dense_1 = nn.Linear(512, SIZE_EMB)
        self.hidden = None
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        h = self.embed(x)
        h.unsqueeze_(0)
        h, self.hidden = self.lstm_1(h, self.hidden)
        # TODO: move loop over sequence inside and don't compute unused dense outputs
        h = self.dense_1(h[0])
        return h

    def _repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self._repackage_hidden(v) for v in h)

    def truncate(self):
        if self.hidden is not None:
            self.hidden = self._repackage_hidden(self.hidden)