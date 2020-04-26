import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from pathlib import Path
from protonn.utils import load_json
import vecto
from .data import load_corpus


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
        # print(h.shape)
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


def init_model(cnt_words):
    net = Net(cnt_words)
    if torch.cuda.is_available():
        net.to("cuda")
    # optimizer = optim.SGD(net.parameters(), 0.01)
    optimizer = optim.Adam(net.parameters())
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    return net, optimizer, scheduler


def load_model(path):
    path = Path(path)
    params = load_json(path / "metadata.json")
    checkpoint = torch.load(path / "model_last.pkl")
    vocab = vecto.vocabulary.Vocabulary()
    vocab.load(path / "vocab")
    corpus_ids = load_corpus(params["path_corpus"], vocab)
    net, optimizer, scheduler = init_model(vocab.cnt_words)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return net, optimizer, scheduler, params, vocab, corpus_ids
