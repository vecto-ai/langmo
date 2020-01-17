import torch
import numpy as np
import vecto.vocabulary


def load_corpus(path_corpus):
    print("creating vocab")
    vocab = vecto.vocabulary.create_from_file(path_corpus)
    corpus_ids = vecto.corpus.load_file_as_ids(path_corpus, vocab)
    corpus_ids = corpus_ids.astype(np.int64)[:4500]
    corpus_ids.shape
    print(len(corpus_ids), max(corpus_ids))
    corpus_ids = torch.tensor(corpus_ids)
    if torch.cuda.is_available():
        corpus_ids = corpus_ids.to("cuda")
    return vocab, corpus_ids
