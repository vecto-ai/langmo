import torch.nn as nn


class Net(nn.Module):
    def __init__(self, size_vocab, size_embedding):
        super().__init__()
        self.emb_in = nn.Embedding(size_vocab, size_embedding)
        # self.emb_out = nn.Embedding(size_vocab, size_embedding)
        initrange = 0.1
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        # self.emb_out.weight.data.uniform_(-initrange, initrange)
        self.out = nn.Linear(size_embedding, size_vocab)

    def forward(self, center, context):
        emb_in = self.emb_in(center)
        # emb_out = self.emb_out(context)
        res = self.out(emb_in)
        # print(res.shape)
        # res = emb_out * emb_in
        # norm_in = torch.norm(emb_in, 2, 1)
        # norm_out = torch.norm(emb_out, 2, 2)
        # norm = norm_out * norm_in
        # res = res.sum(axis=2)
        # res /= norm
        # TODO: loss to separate class?
        return res
