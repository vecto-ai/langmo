import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import pandas
import numpy as np

import yaml
import os
import sys
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoConfig


def read_ds(path, tokenizer):
    train = []
    cnt = 0
    df = pandas.read_csv(path,
                         sep="\t",
                         header=0,
                         quoting=3,
                         keep_default_na=False)
    dic_labels = {l: i for i, l in enumerate(sorted(df["gold_label"].unique()))}
    #df["sentence1"] = df["sentence1"].apply(lambda s: s.lower())
    #df["sentence2"] = df["sentence2"].apply(lambda s: s.lower())
#    print(df["sentence1"][:10])
    print(dic_labels)
    sent1 = map(lambda x: x[:64], df["sentence1"])
    sent2 = map(lambda x: x[:64], df["sentence2"])
    sent1 = map(tokenizer.encode, sent1)
    sent2 = map(tokenizer.encode, sent1)
    labels = map(lambda x: dic_labels[x], df["gold_label"])
    tuples = zip(zip(sent1, sent2), labels)
    return tuples


# TODO: make it actually an iterator
class Iterator:
    def __init__(self, tuples_train, size_batch):
        self.size_batch = size_batch
        train = sorted(tuples_train, key=lambda x: len(x[0][0]))
        self.cnt_samples = len(train)
        self.batches = []
        for i in range(0, len(train), size_batch):
            batch = train[i:i + size_batch]
            self.batches.append(self.zero_pad_batch(batch))

    def zero_pad_item(self, sample, max_len):
        size_pad = max_len - len(sample)
        assert size_pad >= 0
        res = np.hstack([sample, np.zeros(size_pad, dtype=np.int64)])
        return res

    def zero_pad_batch(self, batch):
        max_len = max([len(i[0][0]) for i in batch])
        list_sents = []
        list_masks = []
        list_segments = []
        list_labels = []
        for sample in batch:
            (sent, segments), label = sample
            mask = [1] * len(sent)
            sent = self.zero_pad_item(sent, max_len)
            segments = self.zero_pad_item(segments, max_len)
            mask = self.zero_pad_item(mask, max_len)
            list_sents.append(sent)
            list_masks.append(mask)
            list_segments.append(segments)
            list_labels.append(label)
        block_sent = np.vstack(list_sents)
        block_masks = np.vstack(list_masks)
        block_segments = np.vstack(list_segments)
        # block_s1 = np.rollaxis(block_s1, 1, start=0)  # make it sequence-first
        labels = np.array(list_labels)
        return (block_sent, block_masks, block_segments), labels


class ModelHans(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, **kwargs):
        loss, logits = self.net(**kwargs)
        entail = logits[:, 1:2]
        non_entail = torch.cat((logits[:, 0:1], logits[:, 2:3]), 1)
        non_entail = non_entail.max(axis=1)
        new_logits = torch.cat((entail, non_entail.values.unsqueeze(1)), 1)
        return 0, new_logits


def train_batch(net, optimizer, batch, train):
    (ids, mask, segments), labels = batch
    ids = torch.from_numpy(ids)
    mask = torch.from_numpy(mask)
    segments = torch.from_numpy(segments)
    labels = torch.from_numpy(labels)
    # print(labels)
    ids = ids.to("cuda")
    mask = mask.to("cuda")
    segments = segments.to("cuda")
    labels = labels.to("cuda")
    loss, logits = net(input_ids=ids,
                       attention_mask=mask,
                       token_type_ids=segments,
                       labels=labels)
    if train:
        net.zero_grad()
        loss.backward()
        optimizer.step()
    max_index = logits.max(dim=1)[1]
    #print(max_index)
    mask_correct = max_index == labels
    cnt_correct = mask_correct.sum()
    return float(loss), int(cnt_correct)


def train_epoch(net, optimizer, scheduler, iterator, params, train):
    losses = []
    if train:
        net.train()
    else:
        net.eval()
    cnt_correct = 0
    for batch in iterator.batches:
        loss, correct_batch = train_batch(net, optimizer, batch, train)
        losses.append(loss)
        cnt_correct += correct_batch
    #if train:
    #    scheduler.step()
    return np.mean(losses), cnt_correct / iterator.cnt_samples


def make_iter(path, tokenizer, size_batch=32):
    train_tuples = read_ds(path, tokenizer)
    train_tuples = list(train_tuples)
    sentpairs, labels = zip(*train_tuples)
    sent_merged = [a + b[1:] for a, b in sentpairs]
    segment_ids = [[0] * len(a) + [1] * (len(b) - 1) for a, b in sentpairs]
    inputs = list(zip(sent_merged, segment_ids))
    tuples_merged = list(zip(inputs, labels))
    it = Iterator(tuples_merged, size_batch)
    return it


def main():
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    params["path_train"] = os.path.join(params["path_mnli"], "train.tsv")
    params["path_val"] = os.path.join(params["path_mnli"], "dev_matched.tsv")
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    it_train = make_iter(params["path_train"], tokenizer, params["size_batch"])
    it_val = make_iter(params["path_val"], tokenizer, params["size_batch"])
    it_hans = make_iter(os.path.join(params["path_hans"], "heuristics_evaluation_set.txt"),
                        tokenizer,
                        params["size_batch"])
    for i in range(6):
        ids = it_hans.batches[0][0][0][i][:10]
        s = tokenizer.decode(ids)
        print(it_hans.batches[0][1][i], s)
    config = AutoConfig.from_pretrained(
        "albert-base-v2",
        num_labels=3)#,
        #finetuning_task=data_args.task_name,
        #cache_dir=model_args.cache_dir)

    model_classifier = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", config=config)
    model_classifier.to("cuda")
    model_hans = ModelHans(model_classifier)
    optimizer = optim.Adam([param for param in model_classifier.parameters() if param.requires_grad == True], lr=0.00001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    params["train_log"] = []
    for id_epoch in range(params["cnt_epochs"]):
        loss, acc = train_epoch(model_classifier, optimizer, scheduler, it_train, params, True)
        epoch_stats = {}
        epoch_stats["id"] = id_epoch
        epoch_stats["loss"] = loss
        epoch_stats["acc"] = acc
        epoch_stats["lr"] = optimizer.param_groups[0]['lr']
        params["train_log"].append(epoch_stats)
        val_loss, val_acc = train_epoch(model_classifier, optimizer, scheduler, it_val, params, False)
        epoch_stats["val_loss"] = val_loss
        epoch_stats["val_acc"] = val_acc
        val_loss, val_acc = train_epoch(model_hans, optimizer, scheduler, it_hans, params, False)
        epoch_stats["val_loss_hans"] = val_loss
        epoch_stats["val_acc_hans"] = val_acc
        print(id_epoch,
              f"loss: {params['train_log'][-1]['loss']:.4f}",
              f"acc: {params['train_log'][-1]['acc']:.4f}",
              f"val_loss: {params['train_log'][-1]['val_loss']:.4f}",
              f"val_acc: {params['train_log'][-1]['val_acc']:.4f}",
              f"hans_acc: {params['train_log'][-1]['val_acc_hans']:.4f}",
              f"lr: {params['train_log'][-1]['lr']}",
              # f"time ep: {time_end - time_start:.3f}s",
              # f"time total: {datetime.timedelta(seconds=time_total)}",
              )


if __name__ == '__main__':
    main()
