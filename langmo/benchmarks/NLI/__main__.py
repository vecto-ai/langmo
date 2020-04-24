import sys
import os
import yaml
import datetime
import torch
import torch.optim as optim
import vecto
import vecto.embeddings
import numpy as np
import torch.nn.functional as F
from protonn.utils import save_data_json
from langmo.utils import get_unique_results_path
from .data import Iterator, read_ds
from .model import Net
from timeit import default_timer as timer


def make_snapshot(net, optimizer, scheduler, id_epoch, params):
    # print(f"creating ep {id_epoch} snapshot")
    net.cpu()
    net.hidden = None
    save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
    # vocab.save_to_dir(os.path.join(params["path_results"], "vocab"))
    name_snapshot = f"snap_ep_{id_epoch:03}"
    # schedule_eval_script(command_eval)

    torch.save({'epoch': id_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(params["path_results"], "model_last.pkl"))

    # scheduler_state_dict': scheduler.state_dict()},


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
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    path_results_base = "./out/NLI"
    params["path_results"] = get_unique_results_path(path_results_base)
    save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
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
    params["train_log"] = []
    print("start training")
    params["time_start_training"] = timer()
    for id_epoch in range(params["cnt_epochs"]):
        loss, acc = train_epoch(net, optimizer, it_train)
        time_end = timer()
        time_total = (time_end - params["time_start_training"])
        loss_val, acc_val = train_epoch(net, optimizer, it_val, False)
        epoch_stats = {}
        epoch_stats["loss"] = loss
        epoch_stats["accuracy"] = acc
        epoch_stats["val_loss"] = loss_val
        epoch_stats["val_accuracy"] = acc_val
        params["train_log"].append(epoch_stats)
        print(id_epoch,
              f"loss: {loss:.4f}",
              f"acc: {acc:.4f}",
              f"loss_val: {loss_val:.4f}",
              f"acc_val: {acc_val:.4f}",
              f"time total: {datetime.timedelta(seconds=time_total)}")
        # TODO: log to fs
        scheduler = None
        make_snapshot(net, optimizer, scheduler, id_epoch, params)


if __name__ == "__main__":
    main()
