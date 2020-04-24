import sys
import yaml
import torch
import torch.optim as optim
import vecto
import vecto.embeddings
import numpy as np
import torch.nn.functional as F
from .data import Iterator, read_ds
from .model import Net


def train_batch(net, optimizer, batch, train):
    s1, s2 = batch[0]
    target = batch[1]
    net.zero_grad()
    s1 = torch.from_numpy(s1)
    s2 = torch.from_numpy(s2)
    target = torch.from_numpy(target)
    s1 = s1.to("cuda")
    s2 = s2.to("cuda")
    target = target.to("cuda")
    logits = net(s1, s2)
    # print(logits.shape)
    loss = F.cross_entropy(logits, target)
    if train:
        loss.backward()
        optimizer.step()
    max_index = logits.max(dim=1)[1]
    mask_correct = max_index == target
    cnt_correct = mask_correct.sum()
    return float(loss), int(cnt_correct)


def train_epoch(net, optimizer, iter, train=True):
    net.to("cuda")
    losses = []
    cnt_correct = 0
    for i in range(len(iter.batches)):
        loss, correct_batch = train_batch(net, optimizer, iter.batches[i], train)
        losses.append(loss)
        cnt_correct += correct_batch
    return np.mean(losses), cnt_correct / iter.cnt_samples


def main():
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        return
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg)
    embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
    print("loaded embeddings")
    net = Net(embs)
    print("constructed a model")
    batch_size = 4
    train_tuples = read_ds(params["path_train"], embs, params["test"])
    val_tuples = read_ds(params["path_val"], embs, params["test"])
    it_train = Iterator(train_tuples, batch_size)
    it_val = Iterator(val_tuples, batch_size)
    optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad == True], lr=0.001)
    for id_epoch in range(params["cnt_epochs"]):
        loss, acc = train_epoch(net, optimizer, it_train)
        loss_val, accuracy_val = train_epoch(net, optimizer, it_val, False)
        print(loss, acc, loss_val, accuracy_val)
        # TODO: log to fs


if __name__ == "__main__":
    main()
