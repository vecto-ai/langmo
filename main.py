import os
import random
from pathlib import Path
import datetime
import vecto.corpus
import vecto.embeddings.dense
from vecto.embeddings.dense import WordEmbeddingsDense
import vecto.benchmarks
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from model import Net
from protonn.utils import get_time_str
from protonn.utils import save_data_json, load_json
import platform
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data import load_corpus


pos_corpus = 0
cnt_epochs = 4000
offset_negative = 2000
offset_negative_max_random_add = 100


def init_model(cnt_words):
    net = Net(cnt_words)
    if torch.cuda.is_available():
        net.to("cuda")
    # optimizer = optim.SGD(net.parameters(), 0.01)
    optimizer = optim.Adam(net.parameters(), 0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    return net, optimizer, scheduler


def make_snapshot(net, optimizer, scheduler, id_epoch, vocab, params):
    print(f"creating ep {id_epoch} snapshot")
    save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
    embeddings = WordEmbeddingsDense()
    embeddings.vocabulary = vocab
    embeddings.metadata["vocabulary"] = vocab.metadata
    embeddings.metadata["vsmlib_version"] = vecto.__version__
    embeddings.metadata["cnt_epochs"] = id_epoch
    embeddings.metadata.update(params)
    embeddings.matrix = net.embed.weight.data.cpu().numpy()
    name_snapshot = f"snap_ep_{id_epoch}"
    embeddings.save_to_dir(os.path.join(params["path_results"], name_snapshot, "embs"))

    torch.save({'epoch': id_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
                os.path.join(params["path_results"], "model_last.pkl"))


# TODO: write trainer or use existing lib
def train_epoch(corpus_ids, optimizer, net, params):
    global pos_corpus
    pos_corpus = 0
    losses_epoch = []
    # TODO: iterate oven number of batches with non-sequential sampling
    max_pos_corpus = corpus_ids.shape[0] - params["len_sequence"] - offset_negative - offset_negative_max_random_add
    if pos_corpus > max_pos_corpus:
        RuntimeError("training corpus too short")
    while pos_corpus < max_pos_corpus:
        losses_epoch.append(train_batch(corpus_ids, optimizer, net, params))
    return np.mean(losses_epoch)


def train_batch(corpus_ids, optimizer, net, params):
    global pos_corpus
    batch_size = params["batch_size"]
    optimizer.zero_grad()
    for _ in range(params["len_sequence"]):
        # TODO: sample sequences from different parts of the corpus
        batch = corpus_ids[pos_corpus:pos_corpus + batch_size]
        predicted = net(batch)
        pos_corpus += 1
    targets_positive = corpus_ids[pos_corpus: pos_corpus + batch_size]
    loss_positive = F.cosine_similarity(predicted, net.embed(targets_positive)).sum()
    pos_start_negative = pos_corpus + offset_negative + random.randint(0, offset_negative_max_random_add)
    targets_negative = corpus_ids[pos_start_negative: pos_start_negative + batch_size]
    loss_negative = - F.cosine_similarity(predicted, net.embed(targets_negative)).sum()
    loss = loss_positive + loss_negative
    loss.backward()
    # loss.unchain_backward()
    optimizer.step()
    return float(loss.data)


def load_model(path):
    path = Path(path)
    params = load_json(path / "metadata.json")
    checkpoint = torch.load(path / "model_last.pkl")
    vocab, corpus_ids = load_corpus(params["path_corpus"])
    net, optimizer, scheduler = init_model(vocab.cnt_words)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return net, optimizer, scheduler, params, vocab, corpus_ids


def main():
    # TODO: decide init or load by args or presence of metadata file
    if False:  # init vs load
        params = {}
        hostname = platform.node()
        if hostname.endswith("titech.ac.jp"):
            path_results_base = "/work/alex/data/DL_outs/NLP/embed_ptoto1"
        else:
            path_results_base = "./out"
        params["batch_size"] = 4
        params["loss_history"] = []
        params["len_sequence"] = 12
        params["path_corpus"] = "./corpus/brown.txt"
        vocab, corpus_ids = load_corpus(params["path_corpus"])
        net, optimizer, scheduler = init_model(vocab.cnt_words)
        params["path_results"] = os.path.join(path_results_base, f"{get_time_str()}_{hostname}")
        os.makedirs(params["path_results"], exist_ok=True)
        params["time_start_training"] = timer()
        make_snapshot(net, optimizer, scheduler, 0, vocab, params)
    else:  # load
        path_load = "/home/blackbird/Projects_heavy/NLP/langmo/out/20.01.17_11.31.56_cornus"
        net, optimizer, scheduler, params, vocab, corpus_ids = load_model(path_load)

    print("training")
    for id_epoch in range(len(params["loss_history"]), cnt_epochs):
        time_start = timer()
        loss_epoch = train_epoch(corpus_ids, optimizer, net, params)
        params["loss_history"].append(loss_epoch)
        time_end = timer()
        time_total = (time_end - params["time_start_training"])
        print(id_epoch,
              f"loss: {params['loss_history'][-1]:.4f}",
              f"lr: {optimizer.param_groups[0]['lr']:.5f}",
              f"time ep: {time_end - time_start:.3f}s",
              f"time total: {datetime.timedelta(seconds=time_total)}",
              )
        plt.plot(np.arange(len(params["loss_history"])), params["loss_history"])
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.savefig(os.path.join(params["path_results"], "loss.pdf"))
        plt.clf()
        id_epoch += 1
        make_snapshot(net, optimizer, scheduler, id_epoch, vocab, params)
        scheduler.step()


if __name__ == "__main__":
    main()
