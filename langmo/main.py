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
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from protonn.utils import get_time_str
from protonn.utils import save_data_json
import platform
import torch
import torch.nn.functional as F
from .model import Net, init_model, load_model
from .data import load_corpus
from .generate import generate


pos_corpus = 0
cnt_epochs = 300


# TODO: move this to protonn
def schedule_eval_script(command):
    path_scripts = "/tmp/protonn/scripts"
    os.makedirs(path_scripts, exist_ok=True)
    unique_name = uuid.uuid4().hex + ".sh"
    path_script = os.path.join(path_scripts, unique_name)
    with open(path_script, "w") as f:
        f.write("path_scrypt=$(pwd)/$0\n")
        f.write("set -e\n")
        f.write(command)
        f.write("\nrm $path_scrypt\n")
    os.chmod(path_script, 0o766)
    # TODO: schedule to job queue


def make_snapshot(net, optimizer, scheduler, id_epoch, vocab, params):
    print(f"creating ep {id_epoch} snapshot")
    net.cpu()
    net.hidden = None
    save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
    vocab.save_to_dir(os.path.join(params["path_results"], "vocab"))
    embeddings = WordEmbeddingsDense()
    embeddings.vocabulary = vocab
    embeddings.metadata.update(params)
    embeddings.metadata["vocabulary"] = vocab.metadata
    embeddings.metadata["cnt_epochs"] = id_epoch
    embeddings.metadata.update(params)
    embeddings.matrix = net.embed.weight.data.cpu().numpy()
    name_snapshot = f"snap_ep_{id_epoch:03}"
    path_embeddings = os.path.join(params["path_results"], name_snapshot, "embs")
    embeddings.save_to_dir(path_embeddings)
    path_eval_results = os.path.join(params["path_results"], name_snapshot, "eval")
    path_this_module = Path(__file__).parent.parent
    command_eval = f"cd {path_this_module}\npython3 -m langmo.evaluate {path_embeddings} {path_eval_results}"
    # schedule_eval_script(command_eval)

    torch.save({'epoch': id_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
                os.path.join(params["path_results"], "model_last.pkl"))
    embeddings.normalize()
    seeds = ["the meaning of life is",
             "once upon a time",
             "this is the poing of my proposal",
             "man is to woman as king is to"]

    with open(os.path.join(params["path_results"], name_snapshot, "generated.txt"), "w") as f:
        for seed in seeds:
            generated = generate(seed, net, embeddings, vocab)
            f.write(f"seed: {seed}\n")
            f.write(f"generated {generated}\n\n")
    net.hidden = None
    if torch.cuda.is_available():
        net.to("cuda")


# TODO: write trainer or use existing lib
def train_epoch(corpus_ids, optimizer, net, params):
    global pos_corpus
    pos_corpus = 0
    losses_epoch = []
    # TODO: iterate oven number of batches with non-sequential sampling
    max_pos_corpus = corpus_ids.shape[0] \
        - params["len_sequence"] \
        - params["offset_negative"] \
        - params["offset_negative_max_random_add"]
    if pos_corpus > max_pos_corpus:
        RuntimeError("training corpus too short")
    while pos_corpus < max_pos_corpus:
        losses_epoch.append(train_batch(corpus_ids, optimizer, net, params))
    return np.mean(losses_epoch)


def train_batch(corpus_ids, optimizer, net, params):
    global pos_corpus
    offset_negative = params["offset_negative"]
    batch_size = params["batch_size"]
    net.zero_grad()
    # unchain properly here
    # net.hidden = None
    net.truncate()
    # print(pos_corpus)
    for _ in range(params["len_sequence"]):
        # TODO: sample sequences from different parts of the corpus
        batch = corpus_ids[pos_corpus:pos_corpus + batch_size]
        predicted = net(batch)
        #print(predicted)
        #exit(-1)
        pos_corpus += 1
    targets_positive = corpus_ids[pos_corpus: pos_corpus + batch_size]
    loss_positive = - F.cosine_similarity(predicted, net.embed(targets_positive)).sum()
    pos_start_negative = pos_corpus + offset_negative + random.randint(0, params["offset_negative_max_random_add"])
    targets_negative = corpus_ids[pos_start_negative: pos_start_negative + batch_size]
    loss_negative = F.cosine_similarity(predicted, net.embed(targets_negative)).sum()
    loss = loss_positive + loss_negative
    loss.backward()
    # loss.unchain_backward()
    optimizer.step()
    return float(loss.data)


def main():
    if len(sys.argv) == 1:
        params = {}
        hostname = platform.node()
        if hostname.endswith("titech.ac.jp"):
            path_results_base = "/work/alex/data/DL_outs/NLP/embed_proto1"
        else:
            path_results_base = "./out"
        params["batch_size"] = 4
        params["loss_history"] = []
        params["len_sequence"] = 12
        params["offset_negative"] = 2000
        params["offset_negative_max_random_add"] = 100
        # params["path_corpus"] = "/work/data/NLP/corpora/raw_texts/Eng/BNC/bnc.txt.gz"
        params["path_corpus"] = "./corpus/"
        params["vecto_version"] = vecto.__version__
        # TODO: cmd arg for load vocab
        vocab = vecto.vocabulary.create_from_path(params["path_corpus"], min_frequency=10)
        corpus_ids = load_corpus(params["path_corpus"], vocab)
        net, optimizer, scheduler = init_model(vocab.cnt_words)
        params["path_results"] = os.path.join(path_results_base, f"{get_time_str()}_{hostname}")
        os.makedirs(params["path_results"], exist_ok=True)
        params["time_start_training"] = timer()
        print("pre-heating for bs")
        batch = corpus_ids[0: params["batch_size"]]
        predicted = net(batch)
        make_snapshot(net, optimizer, scheduler, 0, vocab, params)
    else:  # load
        print("resuming")
        path_load = sys.argv[1]
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
