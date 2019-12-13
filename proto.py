import os
import random
import datetime
import vecto.corpus
import vecto.vocabulary
import vecto.embeddings.dense
from vecto.embeddings.dense import WordEmbeddingsDense
import vecto.benchmarks
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from model import Net
from protonn.utils import get_time_str
import platform
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


params = {}
hostname = platform.node()
if hostname.endswith("titech.ac.jp"):
    path_results_base = "/work/alex/data/DL_outs/NLP/embed_ptoto1"
else:
    path_results_base = "./out"
params["path_results"] = os.path.join(path_results_base, f"{get_time_str()}_{hostname}")
os.makedirs(params["path_results"], exist_ok=True)

path_corpus = "./corpus/brown.txt"

vocab = vecto.vocabulary.create_from_file(path_corpus)
corpus_ids = vecto.corpus.load_file_as_ids(path_corpus, vocab)
corpus_ids = corpus_ids.astype(np.int64)
corpus_ids.shape
print(len(corpus_ids), max(corpus_ids))

corpus_ids = torch.tensor(corpus_ids)
#corpus_ids = corpus_ids.to("cuda")
net = Net(vocab.cnt_words)
#net.to("cuda")

# optimizer = optim.SGD(net.parameters(), 0.01)
optimizer = optim.Adam(net.parameters(), 0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

loss_history = []
pos_corpus = 0

cnt_corpus_passes = 0
batch_size = 4
len_sequence = 12
cnt_batches_per_epoch = 256
cnt_epochs = 4000
offset_negative = 2000
offset_negative_max_random_add = 100


def make_snapshot():
    embeddings = WordEmbeddingsDense()
    embeddings.vocabulary = vocab
    embeddings.metadata["vocabulary"] = vocab.metadata
    embeddings.metadata["vsmlib_version"] = vecto.__version__
    embeddings.metadata["cnt_epochs"] = cnt_corpus_passes
    embeddings.metadata.update(params)
    embeddings.matrix = net.embed.weight.data.cpu().numpy()
    name_snapshot = f"snap_ep_{cnt_corpus_passes}"
    embeddings.save_to_dir(os.path.join(params["path_results"], name_snapshot))


def train_sequence():
    global pos_corpus
    global cnt_corpus_passes
    optimizer.zero_grad()
    if pos_corpus > corpus_ids.shape[0] - len_sequence - offset_negative - offset_negative_max_random_add:
        print("reseting corpus pointer to start")
        pos_corpus = 0
        cnt_corpus_passes += 1
        make_snapshot()
        scheduler.step()
    for _ in range(len_sequence):
        # TODO: sample sequences from different parts of the corpus
        batch = corpus_ids[pos_corpus:pos_corpus + batch_size]
        predicted = net(batch)
        pos_corpus += 1
    targets_positive = corpus_ids[pos_corpus: pos_corpus + batch_size]
    loss_positive = F.cosine_similarity(predicted,  net.embed(targets_positive)).sum()
    pos_start_negative = pos_corpus + offset_negative + random.randint(0, offset_negative_max_random_add)
    targets_negative = corpus_ids[pos_start_negative: pos_start_negative + batch_size]
    loss_negative = - F.cosine_similarity(predicted, net.embed(targets_negative)).sum()
    loss = loss_positive + loss_negative
    loss.backward()
    # loss.unchain_backward()
    optimizer.step()
    return float(loss.data)


make_snapshot()
time_start_training = timer()

for id_epoch in range(cnt_epochs):
    time_start = timer()
    loss_epoch = 0
    for id_batch in range(cnt_batches_per_epoch):
        loss_epoch += train_sequence()
    loss_epoch /= cnt_batches_per_epoch
    loss_history.append(loss_epoch)
    time_end = timer()
    print(cnt_corpus_passes,
          id_epoch,
          pos_corpus,
          f"loss: {loss_history[-1]:.4f}",
          f"lr: {optimizer.param_groups[0]['lr']:.5f}",
          f"time ep: {time_end - time_start:.3f}s",
          f"time total: {datetime.timedelta(seconds=(time_end - time_start_training))}",
          )
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.savefig(os.path.join(params["path_results"], "loss.pdf"))
    plt.clf()


# ---------------------evalutation --------------
# embeddings = WordEmbeddingsDense()
# embeddings.matrix = net.embed.W.data
# embeddings.vocabulary = vocab
# #embeddings.metadata = {}

# embeddings.get_most_similar_words("man")


# from vecto.benchmarks.similarity import Similarity
# my_similarity = Similarity()

# my_similarity.run(embeddings, "/mnt/work/Data/similarity")
