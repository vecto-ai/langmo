import numpy as np
import sys
import os
import yaml
import torch
import torch.optim as optim
import datetime
import vecto.vocabulary
from vecto.embeddings.dense import WordEmbeddingsDense
from protonn.utils import save_data_json
from timeit import default_timer as timer
from langmo.utils import get_unique_results_path
from .model import Net, W2V_NS
from .data import DirWindowIterator
from langmo.evaluate import report_neigbours


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
    sim_man_woman = embeddings.cmp_words("man", "woman")
    print("sim_man_woman", sim_man_woman)
    name_snapshot = f"snap_ep_{id_epoch:03}"
    path_embeddings = os.path.join(params["path_results"], name_snapshot, "embs")
    embeddings.save_to_dir(path_embeddings)
    embeddings.cache_normalized_copy()
    report_neigbours(embeddings, os.path.join(path_embeddings, "eval/neighbours"))
    # path_eval_results = os.path.join(params["path_results"], name_snapshot, "eval")
    # path_this_module = Path(__file__).parent.parent
    if torch.cuda.is_available():
        net.to("cuda")


def train_batch(net, optimizer, batch):
    center, context = batch
    print(center)
    context = np.rollaxis(context, 1, start=0)
    center = torch.from_numpy(center)
    context = torch.from_numpy(context)
    center = center.to("cuda")
    context = context.to("cuda")
    net.zero_grad()
    loss = net(center, context)
    loss.backward()
    optimizer.step()
    return float(loss)


def train_epoch(id_epoch, net, optimizer, it, vocab, params):
    time_start = timer()
    losses_epoch = []
    while True:
        batch = next(it)
        loss = train_batch(net, optimizer, batch)
        losses_epoch.append(loss)
        if it.is_new_epoch:
            break
    make_snapshot(net, id_epoch, vocab, params)
    time_end = timer()
    elapsed_sec = time_end - time_start
    elapsed_str = datetime.timedelta(seconds=elapsed_sec)
    loss_ep = np.mean(losses_epoch)
    mean_emb = float(net.emb_in.weight.data.mean())
    std_emb = float(net.emb_in.weight.data.mean())
    print(f"{id_epoch} loss: {loss_ep:.3}, mean: {mean_emb:.3}, std: {std_emb:.3}", elapsed_str)
    # print(net.emb_in.weight.data[:3])


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
    assert vocab.cnt_words > 1
    # net = Net(vocab.cnt_words, 128)
    net = W2V_NS(vocab.cnt_words, params["embedding_size"], params)
    net.cuda()
    optimizer = optim.SGD([param for param in net.parameters() if param.requires_grad is True],
                          lr=0.01)

    print(vocab.cnt_words)
    it = DirWindowIterator(params["path_corpus"],
                           vocab,
                           params["window_size"],
                           params["batch_size"],
                           language='eng',
                           repeat=True)
    for i in range(params["cnt_epochs"]):
        train_epoch(i, net, optimizer, it, vocab, params)


if __name__ == "__main__":
    main()
