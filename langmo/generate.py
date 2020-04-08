import os
import random
import sys
import uuid
from pathlib import Path
import datetime
import vecto.corpus
import vecto.embeddings.dense
from vecto.embeddings.dense import WordEmbeddingsDense
import vecto.benchmarks
import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from .model import Net, load_model
import torch


# TODO: insert UNK token
def generate(seed, net, embeddings, vocab):
    for t in seed.split():
        i = vocab.get_id(t)
        # print(t, i)
        pred = net(torch.tensor([i]))
        # print(pred)
    generated = []
    for i in range(30):
        vec = pred.data.numpy()[0]
        # print(vec.shape, vec[:10])
        pred_word = embeddings.get_most_similar_words(vec)[0][0]
        generated.append(pred_word)
        id_next = vocab.get_id(pred_word)
        pred = net(torch.tensor([id_next]))
    return generated


def generate_all(net, vocab):
    embeddings = WordEmbeddingsDense()
    embeddings.vocabulary = vocab
    embeddings.matrix = net.embed.weight.data.cpu().numpy()
    embeddings.normalize()
    print("generating")
    seed = "the meaning of life is"
    seed = "once upon a time"
    generated = generate(seed, net, embeddings, vocab)
    print("seed:", seed)
    print("generated", generated)


def main():
    print("resuming")
    path_load = sys.argv[1]
    net, optimizer, scheduler, params, vocab, corpus_ids = load_model(path_load)
    net.cpu()
    generate_all(net, vocab)


if __name__ == "__main__":
    main()
