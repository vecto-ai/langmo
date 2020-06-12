import json
import pandas
import numpy as np
import torch
import torch.optim as optim

from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoConfig

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

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
    sent1 = map(tokenizer.encode, df["sentence1"])
    sent2 = map(tokenizer.encode, df["sentence2"])
    labels = map(lambda x: dic_labels[x], df["gold_label"])
    tuples = zip(zip(sent1, sent2), labels)
    return tuples

params = {}
params["cnt_epochs"] = 2
params["path_train"] = "/groups2/gcb50300/chinese_room/subsample_rand/seed_46/12288/train.tsv"
train_tuples = read_ds(params["path_train"], tokenizer)
train_tuples=list(train_tuples)
sentpairs, labels  = zip(*train_tuples)
sent_merged = [a + b[1:] for a,b in sentpairs]
segment_ids = [[0] * len(a) + [1] * (len(b) - 1) for a,b in sentpairs]
inputs = list(zip(sent_merged, segment_ids))
tuples_merged = list(zip(inputs, labels))

# iterator
# TODO: make it actually an iterator
class Iterator:
    def __init__(self, tuples_train, size_batch):
        self.size_batch = size_batch
        # print(tuples_train[0][0])
        # return
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
        #print(batch)
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
        return (block_sent, block_masks, block_segments),  labels

it_train = Iterator(tuples_merged, size_batch=64)

config = AutoConfig.from_pretrained(
    "albert-base-v2",
    num_labels=3)#,
    #finetuning_task=data_args.task_name,
    #cache_dir=model_args.cache_dir)

model_classifier = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", config=config)
model_classifier.to("cuda")
optimizer = optim.Adam([param for param in model_classifier.parameters() if param.requires_grad == True], lr=0.01)
def train_batch(net, batch, train):
    net.zero_grad()
    (ids, mask, segments), labels = batch
    ids = torch.from_numpy(ids)
    mask = torch.from_numpy(mask)
    segments = torch.from_numpy(segments)
    labels = torch.from_numpy(labels)
    ids = ids.to("cuda")
    mask = mask.to("cuda")
    segments = segments.to("cuda")
    labels = labels.to("cuda")
    loss, logits = net(input_ids=ids,
                       attention_mask=mask,
                       token_type_ids=segments,
                       labels=labels)
    if train:
        loss.backward()
        optimizer.step()
    #print(logits.shape)
    #print(logits)
    #print(labels)
    max_index = logits.max(dim=1)[1]
    #print(max_index)
    mask_correct = max_index == labels
    cnt_correct = mask_correct.sum()
    return float(loss), int(cnt_correct)


def train_epoch(iterator):
    losses = []
    cnt_correct = 0
    for batch in iterator.batches:
        loss, correct_batch = train_batch(model_classifier, batch, train=True)
        losses.append(loss)
        cnt_correct += correct_batch
    return np.mean(losses), cnt_correct /  iterator.cnt_samples

for id_epoch in range(params["cnt_epochs"]):
    loss, acc = train_epoch(it_train)
    print(id_epoch, loss, acc)