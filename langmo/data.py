import numpy as np
import torch
import vecto.vocabulary


def load_corpus(path_corpus, vocab):
    print("loading corpus from", path_corpus)
    corpus_ids = vecto.corpus.load_path_as_ids(path_corpus, vocab)
    corpus_ids = corpus_ids.astype(np.int64)
    corpus_ids.shape
    print(len(corpus_ids), max(corpus_ids))
    corpus_ids = torch.tensor(corpus_ids)
    if torch.cuda.is_available():
        corpus_ids = corpus_ids.to("cuda")
    return corpus_ids
