import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import json
import vecto
import vecto.embeddings
from vecto.corpus.tokenization import word_tokenize_txt
from data import Iterator
from model import Net


def get_batch():
    in_seq = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int64)
    in_seq = np.rollaxis(in_seq, 1, start=0)  # make it sequence-first
    labels = np.array([1, 0], dtype=np.int64)
    return in_seq, labels


# TODO: load sequences from the dataset
def gen_plaintext_samples(question):
    # print(question)
    if question["question"] == "cause":
        res1 = question["choice1"] + " " + question["premise"]
        res2 = question["choice2"] + " " + question["premise"]
    else:
        res1 = question["premise"] + " " + question["choice1"]
        res2 = question["premise"] + " " + question["choice2"]
    labels = np.zeros(2, dtype=np.int64)
    labels[question["label"]] = 1
    return [res1, res2], labels


def train_batch(net, optimizer, in_seq, label, train=True):
    net.zero_grad()
    net.hidden = None
    logits = net(in_seq)
    # loss = F.binary_cross_entropy(pred, label)
    loss = F.cross_entropy(logits, label)
    if train:
        loss.backward()
        optimizer.step()
    max_index = logits.max(dim=1)[1]
    mask_correct = max_index == label
    cnt_correct = mask_correct.sum()
    return float(loss.data), int(cnt_correct.data)


def do_epoch(net, optimizer, it, train=True):
    losses_epoch = []
    cnt_correct_epoch = 0
    for i in range(len(it.batches)):
        in_seq, labels = it.batches[i]
        in_seq = torch.from_numpy(in_seq)
        labels = torch.from_numpy(labels)
        if torch.cuda.is_available():
            in_seq = in_seq.to("cuda")
            labels = labels.to("cuda")
        loss, cnt_correct = train_batch(net, optimizer, in_seq, labels, train)
        losses_epoch.append(loss)
        cnt_correct_epoch += cnt_correct
    return np.mean(losses_epoch), cnt_correct_epoch / it.cnt_samples


def load_dataset(path, vocabulary):
    with open(path) as f:
        train = [json.loads(l) for l in f]
    train = [gen_plaintext_samples(q) for q in train]
    samples, labels = zip(*train)
    samples = [item for sublist in samples for item in sublist]
    labels = np.hstack(labels)
    print(labels[0])
    samples = [word_tokenize_txt(s) for s in samples]
    # print(samples[0])
    samples = [vocabulary.tokens_to_ids(s) for s in samples]
    return zip(samples, labels)


def main():
    print("hi")
    path_embeddings = "/mnt/storage/data/NLP/embeddings/6b.wiki_giga"
    embs = vecto.embeddings.load_from_dir(path_embeddings)
    path_train = "/mnt/storage/data/NLP/datasets/comprehension/COPA/train.jsonl"
    path_val = "/mnt/storage/data/NLP/datasets/comprehension/COPA/val.jsonl"
    ds_train = load_dataset(path_train, embs.vocabulary)
    ds_val = load_dataset(path_val, embs.vocabulary)
    size_batch = 4
    it_train = Iterator(ds_train, size_batch)
    it_val = Iterator(ds_val, size_batch)
    print(f"cnt_train: {it_train.cnt_samples}, cnt_val: {it_val.cnt_samples}")
    net = Net(embs=embs, nhid=128, nlayers=2)
    if torch.cuda.is_available():
        net.to("cuda")

    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    #optimizer = optim.Adam(net.parameters(), lr=0.1)
    optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad == True], lr=0.001)
    cnt_epochs = 100
    for i in range(cnt_epochs):
        loss, accuracy = do_epoch(net, optimizer, it_train)
        loss_val, accuracy_val = do_epoch(net, optimizer, it_val, False)
        print(loss, accuracy, loss_val, accuracy_val)


if __name__ == '__main__':
    main()
