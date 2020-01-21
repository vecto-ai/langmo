import torch.nn as nn

SIZE_EMB = 256


class Net(nn.Module):
    def __init__(self, size_vocab):
        super().__init__()
        self.embed = nn.Embedding(size_vocab, SIZE_EMB)
        self.lstm_1 = nn.LSTM(SIZE_EMB, 512)
        self.dense_1 = nn.Linear(512, SIZE_EMB)

    def forward(self, x):
        h = self.embed(x)
        h.unsqueeze_(0)
        h, hidden = self.lstm_1(h)
        # TODO: move loop over sequence inside and don't compute unused dense outputs
        h = self.dense_1(h[0])
        return h
