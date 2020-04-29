import numpy as np
import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
# import torch.nn.functional as F
from vecto.corpus import DirSlidingWindowCorpus
from vecto.corpus.tokenization import DEFAULT_TOKENIZER, DEFAULT_JAP_TOKENIZER
import vecto.vocabulary
from vecto.embeddings.dense import WordEmbeddingsDense
from protonn.utils import save_data_json
from timeit import default_timer as timer
from langmo.utils import get_unique_results_path
# from vecto.vocabulary import Vocabulary


def make_snapshot(net, id_epoch, vocab, params):
    # print(f"creating ep {id_epoch} snapshot")
    net.cpu()
    save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
    vocab.save_to_dir(os.path.join(params["path_results"], "vocab"))
    embeddings = WordEmbeddingsDense()
    embeddings.vocabulary = vocab
    embeddings.metadata.update(params)
    embeddings.metadata["vocabulary"] = vocab.metadata
    embeddings.metadata["cnt_epochs"] = id_epoch
    embeddings.metadata.update(params)
    embeddings.matrix = net.emb_in.weight.data.cpu().numpy()
    name_snapshot = f"snap_ep_{id_epoch:03}"
    path_embeddings = os.path.join(params["path_results"], name_snapshot, "embs")
    embeddings.save_to_dir(path_embeddings)
    # path_eval_results = os.path.join(params["path_results"], name_snapshot, "eval")
    # path_this_module = Path(__file__).parent.parent
    if torch.cuda.is_available():
        net.to("cuda")


class Net(nn.Module):
    def __init__(self, size_vocab, size_embedding):
        super().__init__()
        self.emb_in = nn.Embedding(size_vocab, size_embedding)
        self.emb_out = nn.Embedding(size_vocab, size_embedding)
        initrange = 0.1
        self.emb_in.weight.data.uniform_(-initrange, initrange)
        self.emb_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, center, context):
        emb_in = self.emb_in(center)
        emb_out = self.emb_out(context)
        # print(emb_in.shape, emb_out.shape)
        res = emb_out * emb_in
        norm_in = torch.norm(emb_in, 2, 1)
        norm_out = torch.norm(emb_out, 2, 2)
        norm = norm_out * norm_in
        # res = F.cosine_similarity(emb_in, emb_out)
        res = res.sum(axis=2)
        res /= norm
        # TODO: optional notmalization here
        # TODO: loss to separate class?
        return res


class RingBuffer():
    def __init__(self, shape_batch, cnt_items, max_id=100):
        # self.buf = np.zeros((cnt_items, *shape_batch), dtype=np.int64)
        self.buf = np.random.randint(0,
                                     high=max_id,
                                     size=(cnt_items, *shape_batch),
                                     dtype=np.int64)
        self.pos = 0

    def pop(self):
        self.pos = (self.pos + 1) % self.buf.shape[0]
        return self.buf[self.pos]

    def push(self, data):
        self.pos = (self.pos + 1) % self.buf.shape[0]
        self.buf[self.pos] = data


class DirWindowIterator():
    def __init__(self, path, vocab, window_size, batch_size, language='eng', repeat=True):
        self.path = path
        self.vocab = vocab
        self.window_size = window_size
        self.language = language
        if language == 'jap':
            self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_JAP_TOKENIZER,
                                               left_ctx_size=self.window_size,
                                               right_ctx_size=self.window_size)
        else:
            self.dswc = DirSlidingWindowCorpus(self.path, tokenizer=DEFAULT_TOKENIZER,
                                               left_ctx_size=self.window_size,
                                               right_ctx_size=self.window_size)
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
                                                       left_ctx_size=self.window_size,
                                                       right_ctx_size=self.window_size)
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


def train_batch(net, optimizer, batch, buf_old_context):
    center, context = batch
    context = np.rollaxis(context, 1, start=0)
    buf_old_context.push(context)
    center = torch.from_numpy(center)
    context = torch.from_numpy(context)
    center = center.to("cuda")
    context = context.to("cuda")
    res = net(center, context)
    # print(res.shape)
    # print(res)
    # loss_positive = - torch.sigmoid(res).mean()
    loss_positive = -res.mean()
    context_negative = buf_old_context.pop()
    context = torch.from_numpy(context_negative)
    context = context.to("cuda")
    res = net(center, context)
    loss_negative = res.mean()
    loss = loss_positive + loss_negative
    loss.backward()
    optimizer.step()
    return float(loss)


def train_epoch(id_epoch, net, optimizer, it, buf_old_context, vocab, params):
    time_start = timer()
    losses_epoch = []
    while True:
        batch = next(it)
        loss = train_batch(net, optimizer, batch, buf_old_context)
        losses_epoch.append(loss)
        if it.is_new_epoch:
            break
    make_snapshot(net, id_epoch, vocab, params)
    time_end = timer()
    elapsed_sec = time_end - time_start
    elapsed_str = datetime.timedelta(seconds=elapsed_sec)
    print(np.mean(losses_epoch), elapsed_str)


def main():
    print("hi")
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        return
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    params["path_results"] = get_unique_results_path(params["path_results"])

    # center = torch.zeros((batch_size), dtype=torch.int64)
    # context = torch.ones((window * 2, batch_size), dtype=torch.int64)
    vocab = vecto.vocabulary.load(params["path_vocab"])
    net = Net(vocab.cnt_words, 128)
    net.cuda()
    optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad is True],
                           lr=0.001)

    print(vocab.cnt_words)
    it = DirWindowIterator(params["path_corpus"],
                           vocab,
                           params["window_size"],
                           params["batch_size"],
                           language='eng',
                           repeat=True)
    # print(context)
    size_old_context = 2000
    buf_old_context = RingBuffer((params["window_size"] * 2, params["batch_size"]),
                                 size_old_context)
    for i in range(params["cnt_epochs"]):
        train_epoch(i, net, optimizer, it, buf_old_context, vocab, params)


if __name__ == "__main__":
    main()
