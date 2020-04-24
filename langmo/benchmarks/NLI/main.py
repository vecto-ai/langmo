import pandas
import json
import sys
import yaml
import torch
import torch.optim as optim
import vecto
import vecto.embeddings
from data import Iterator
from model import Net
import numpy as np
import torch.nn.functional as F


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


def read_ds(path, embs, test=False):
    train = []
    cnt = 0
    with open(path) as f:
        for line in f:
            train.append(json.loads(line))
            cnt += 1
            if test and cnt > 127:
                break
    print(f"{len(train)} samples loaded")
    df = pandas.DataFrame(train)
    dic_labels = {l: i for i, l in enumerate(sorted(df["gold_label"].unique()))}
    df["sentence1"] = df["sentence1"].apply(lambda s: s.lower())
    df["sentence2"] = df["sentence2"].apply(lambda s: s.lower())
#    print(df["sentence1"][:10])
    sent1 = map(embs.vocabulary.tokens_to_ids, df["sentence1"])
    sent2 = map(embs.vocabulary.tokens_to_ids, df["sentence2"])
    labels = map(lambda x: dic_labels[x], df["gold_label"])
    tuples = zip(zip(sent1, sent2), labels)
    return tuples


def main():
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg)
    embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
    print("loaded embeddings")
    net = Net(embs)
    print("constructed a model")
    batch_size = 4
    train_tuples = read_ds(params["path_train"], embs)
    val_tuples = read_ds(params["path_val"], embs)
    it_train = Iterator(train_tuples, batch_size)
    it_val = Iterator(val_tuples, batch_size)
    optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad == True], lr=0.001)
    for i in range(20):
        loss, acc = train_epoch(net, optimizer, it_train)
        loss_val, accuracy_val = train_epoch(net, optimizer, it_val, False)
        print(loss, acc, loss_val, accuracy_val)
        # TODO: log to fs


if __name__ == "__main__":
    main()
