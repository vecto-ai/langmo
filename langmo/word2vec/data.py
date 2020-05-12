import numpy as np
# import vecto.vocabulary
from vecto.corpus.tokenization import word_tokenize_txt
from vecto.corpus import DirSlidingWindowCorpus
from vecto.corpus.tokenization import DEFAULT_TOKENIZER, DEFAULT_JAP_TOKENIZER


class FilePairIter:
    def __init__(self, path, vocab, params):
        self.path = path
        self.window_size = params["window_size"]
        self.batch_size = params["batch_size"]
        self.vocab = vocab
        self.freqs = np.ones(vocab.cnt_words, dtype=np.float32)

    def get_sample_proba(self, id_word):
        self.freqs[id_word] += 1
        frac = self.freqs[id_word] / self.freqs.sum()
        prob = (np.sqrt(frac / 0.001) + 1) * (0.001 / frac)
        return prob

    def gen_from_ids(self, ids):
        for i in range(len(ids)):
            for j in range(max(0, i - self.window_size), i):
                yield(ids[j], ids[i])
            for j in range(i + 1, min(len(ids), i + self.window_size + 1)):
                yield(ids[j], ids[i])

    def clean_ids(self, ids):
        for i in range(ids.shape[0]):
            prob = self.get_sample_proba(ids[i])
            if np.random.random() > prob:
                ids[i] = 0
        return ids[ids != 0]

    def gen(self):
        with open(self.path) as f:
            for line in f:
                tokens = word_tokenize_txt(line)
                ids = self.vocab.tokens_to_ids(tokens)
                ids = self.clean_ids(ids)
                yield from self.gen_from_ids(ids)

    def gen_batch(self):
        center = []
        context = []
        for i, j in self.gen():
            center.append(i)
            context.append(j)
            if len(center) >= self.batch_size:
                yield np.array(center, dtype=np.int64), np.array(context, dtype=np.int64)
                center = []
                context = []


class LousyRingBuffer():
    def __init__(self, shape_batch, cnt_items, max_id=1000):
        # self.buf = np.zeros((cnt_items, *shape_batch), dtype=np.int64)
        self.buf = np.random.randint(0,
                                     high=max_id,
                                     size=(cnt_items, *shape_batch),
                                     dtype=np.int64)
        self.pos = 0

    def pop(self):
        res = self.buf[self.pos]
        # self.pos = (self.pos + 1) % self.buf.shape[0]
        return res

    def push(self, data):
        self.buf[self.pos] = data
        self.pos = (self.pos + 1) % self.buf.shape[0]


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
                # print(next_value)
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
