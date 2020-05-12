import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import LousyRingBuffer


class Net(nn.Module):
    def __init__(self, size_vocab, size_embedding, params):
        super().__init__()
        self.emb_in = nn.Embedding(size_vocab, size_embedding)
        initrange = 0.1
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        self.out = nn.Linear(size_embedding, size_vocab)

    def forward(self, center, context):
        emb_in = self.emb_in(center)
        pred = self.out(emb_in)
        size_seq = context.shape[0]
        size_batch = context.shape[1]
        cnt_classes = pred.shape[-1]
        pred = pred.expand(size_seq, size_batch, cnt_classes).reshape(-1, cnt_classes)
        loss = F.cross_entropy(pred, context.flatten(), ignore_index=0)
        return loss


class W2V_NS(nn.Module):
    def __init__(self, size_vocab, size_embedding, params):
        super().__init__()
        self.emb_in = nn.Embedding(size_vocab, size_embedding)
        self.emb_out = nn.Embedding(size_vocab, size_embedding)
        initrange = 0.1
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        self.emb_out.weight.data.uniform_(-initrange, initrange)
        size_old_context = 2000
        self.buf_old_context = LousyRingBuffer((params["window_size"] * 2, params["batch_size"]),
                                               size_old_context,
                                               size_vocab)

    def forward(self, center, context):
        self.buf_old_context.push(context.cpu().numpy())
        emb_in = self.emb_in(center)
        emb_out = self.emb_out(context)
        pred = emb_out * emb_in
        pred = pred.sum(axis=2)
        loss_positive = 1 - torch.sigmoid(pred).mean()
        context_negative = self.buf_old_context.pop()
        context = torch.from_numpy(context_negative)
        context = context.to("cuda")
        emb_out = self.emb_out(context)
        pred = emb_out * emb_in
        pred = pred.sum(axis=2)
        # print(pred.shape)
        loss_negative = torch.sigmoid(pred).mean()
        # print(loss_positive, loss_negative)
        loss = loss_positive + loss_negative
        # norm_in = torch.norm(emb_in, 2, 1)
         # norm_out = torch.norm(emb_out, 2, 2)
        # norm = norm_out * norm_in
        # res = res.sum(axis=2)
        # res /= norm
        # TODO: loss to separate class?
        return loss
