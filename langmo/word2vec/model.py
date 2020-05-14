import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import LousyRingBuffer


class W2V_SM(nn.Module):
    def __init__(self, size_vocab, size_embedding, params):
        super().__init__()
        self.emb_in = nn.Embedding(size_vocab, size_embedding)
        self.emb_out = nn.Linear(size_embedding, size_vocab, bias=False)
        initrange = 0.01
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        self.emb_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, center, context):
        # print("center:", center.shape)
        # print("context:", context.shape)
        emb_context = self.emb_in(context)
        shape = emb_context.shape
        # size_seq = context.shape[0]
        center.unsqueeze_(1)
        center = center.expand(shape[0], shape[1])
        # print("center bcast:", center.shape)
        emb_context = emb_context.reshape((shape[0] * shape[1], shape[2]))
        center = center.reshape((shape[0] * shape[1],))
        #print(emb_context.shape, center.shape)
        # exit(1)
        #loss = self.loss_func(emb_context, center)
        pred = self.emb_out(emb_context)
        loss = F.cross_entropy(pred, center)
        #emb_in = self.emb_in(center)
        #pred = self.emb_out(emb_in)
        #size_seq = context.shape[0]
        #pred = pred.expand(size_seq, -1, -1).flatten(end_dim=1)
        # print(pred.shape, context.shape)
        #loss = F.cross_entropy(pred, context.flatten(), ignore_index=0)
        # loss = 0
        # for i in range(context.shape[0]):
        #     loss += F.cross_entropy(pred, context[i])
        # loss /= context.shape[0]

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
