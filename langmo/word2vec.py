import numpy as np
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
from vecto.corpus import DirSlidingWindowCorpus
from vecto.corpus.tokenization import DEFAULT_TOKENIZER, DEFAULT_JAP_TOKENIZER
import vecto.vocabulary


class Net(nn.Module):
    def __init__(self, size_vocab, size_embedding):
        super().__init__()
        self.emb_in = nn.Embedding(size_vocab, size_embedding)
        self.emb_out = nn.Embedding(size_vocab, size_embedding)

    def forward(self, center, context):
        emb_in = self.emb_in(center)
        emb_out = self.emb_out(context)
        # print(emb_in.shape, emb_out.shape)
        res = emb_in * emb_out
        res = res.sum(axis=2)
        # TODO: optional not    malization here
        # TODO: loss to separate class?
        return res


class DirWindowIterator():
    def __init__(self, path, vocab, window_size, batch_size, language='eng', repeat=True):
        self.path = path
        self.vocab = vocab
        self.window_size = window_size - 1
        self.language = language
        if language == 'jap':
            self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_JAP_TOKENIZER,
                                               left_ctx_size=self.window_size,
                                               right_ctx_size=self.window_size)
        else:
            self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_TOKENIZER,
                                               left_ctx_size=self.window_size, right_ctx_size=self.window_size)
        self.batch_size = batch_size
        self._repeat = repeat
        self.epoch = 0
        self.is_new_epoch = False
        self.cnt_words_total = 1
        self.cnt_words_read = 0
        # logger.debug("created dir window iterator")

    def next_single_sample(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration
        while True:
            try:
                next_value = next(self.dswc)
                self.cnt_words_read += 1
                if self.epoch == 0:
                    self.cnt_words_total += 1
                break
            except StopIteration:
                self.epoch += 1
                self.is_new_epoch = True
                if self.language == 'jap':
                    self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_JAP_TOKENIZER,
                                                       left_ctx_size=self.window_size,
                                                       right_ctx_size=self.window_size)
                else:
                    self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_TOKENIZER,
                                                       left_ctx_size=self.window_size, right_ctx_size=self.window_size)
            if self.epoch > 0 and self.cnt_words_total < 3:
                print("corpus empty")
                raise RuntimeError("Corpus is empty")

        self.center = self.vocab.get_id(next_value['current'])
        self.context = [self.vocab.get_id(w) for w in next_value['context']]

        # append -1 to ensure the size of context are equal
        while len(self.context) < self.window_size * 2:
            self.context.append(0)
        return self.center, self.context

    @property
    def epoch_detail(self):
        return self.cnt_words_read / self.cnt_words_total

    def __next__(self):
        self.is_new_epoch = False
        centers = []
        contexts = []
        for _ in range(self.batch_size):
            center, context = self.next_single_sample()
            centers.append(center)
            contexts.append(context)
        return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


def train_batch(net, optimizer, batch):
    center, context = batch
    context = np.rollaxis(context, 1, start=0)
    center = torch.from_numpy(center)
    context = torch.from_numpy(context)
    center = center.to("cuda")
    context = context.to("cuda")
    res = net(center, context)
    # print(res.shape)
    # print(res)
    loss_positive = - torch.sigmoid(res).sum()
    loss = loss_positive
    loss.backward()
    optimizer.step()
    return float(loss)


def main():
    print("hi")
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        return
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    # center = torch.zeros((batch_size), dtype=torch.int64)
    # context = torch.ones((window * 2, batch_size), dtype=torch.int64)
    vocab = vecto.vocabulary.load(params["path_vocab"])
    net = Net(vocab.cnt_words, 128)
    net.cuda()
    optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad == True], lr=0.001)

    print(vocab.cnt_words)
    it = DirWindowIterator(params["path_corpus"],
                           vocab,
                           params["window_size"],
                           params["batch_size"],
                           language='eng',
                           repeat=True)
    losses_epoch = []
    while not it.is_new_epoch:
        batch = next(it)
        loss = train_batch(net, optimizer, batch)
        losses_epoch.append(loss)
    print(np.mean(losses_epoch))
    # print(context)


if __name__ == "__main__":
    main()
