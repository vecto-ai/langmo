import sys
import yaml
import torch
import vecto
import vecto.embeddings
import torch.nn.functional as F
from langmo.utils import get_unique_results_path
from .data import NLIDataModule
from .model import Net
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from collections import OrderedDict
from pytorch_lightning.metrics.functional import accuracy


class PLModel(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        # self.example_input_array = torch.zeros((1, 4, 128, 128, 128))

    def forward(self, s1, s2):
        # print(s1.shape)
        return self.net(s1, s2)

    def training_step(self, batch, batch_idx):
        s1, s2, target = batch
        # print(type(target))
        # exit(9)
        logits = self(s1, s2)
        loss = F.cross_entropy(logits, target)
        acc = accuracy(logits, target)
        # result.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=False, sync_dist=True)
        #result = OrderedDict({
        #    'loss': loss,
        #    # 'accuracy': acc,
        #})
        return loss

    def validation_step(self, batch, batch_idx):
        s1, s2, target = batch

        logits = self(s1, s2)
        loss = F.cross_entropy(logits, target)
        acc = accuracy(logits, target)
        self.log('val_loss', loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=False, sync_dist=True)
        #result = OrderedDict({
        #    'loss': loss,
        #    # 'accuracy': acc,
        #})
        #return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002)
        # return torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)


def main():
    #path_data = "/groups1/gac50489/datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_small"
    #path_data = "/groups1/gac50489/datasets/cosmoflow_full/cosmoUniverse_2019_05_4parE_tf"
    if len(sys.argv) < 2:
        print("run main.py config.yaml")
        return
    path_config = sys.argv[1]
    with open(path_config, "r") as cfg:
        params = yaml.load(cfg, Loader=yaml.SafeLoader)
    path_results_base = "./out/NLI"
    params["path_results"] = get_unique_results_path(path_results_base)
    params["batch_size"] = 32
    wandb_logger = WandbLogger(project="NLI")
    # wandb_logger.log_hyperparams(config)
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.0001,
    #     patience=5,
    #     verbose=True,
    #     mode="min",
    # )
    # print("create tainer")
    trainer = pl.Trainer(
        gpus=1,
        num_sanity_val_steps=0,
        max_epochs=5,
        distributed_backend="horovod",
        replace_sampler_ddp=False,
        # early_stop_callback=early_stop_callback,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,
    )
    # print("tainer created")

    embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
    data_module = NLIDataModule(params["path_mnli"], embs.vocabulary, batch_size=params["batch_size"])
    model = PLModel(Net(embs))
    print("fit")
    trainer.fit(model, data_module)

# def make_snapshot(net, optimizer, scheduler, id_epoch, params):
#     # print(f"creating ep {id_epoch} snapshot")
#     net.cpu()
#     net.hidden = None
#     save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
#     # vocab.save_to_dir(os.path.join(params["path_results"], "vocab"))
#     name_snapshot = f"snap_ep_{id_epoch:03}"
#     # schedule_eval_script(command_eval)

#     torch.save({'epoch': id_epoch,
#                 'model_state_dict': net.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict()},
#                 os.path.join(params["path_results"], "model_last.pkl"))

    # scheduler_state_dict': scheduler.state_dict()},


# def train_batch(net, optimizer, batch, train):
#     s1, s2 = batch[0]
#     target = batch[1]
#     if train:
#         net.train()
#     else:
#         net.eval()
#     net.zero_grad()
#     s1 = torch.from_numpy(s1)
#     s2 = torch.from_numpy(s2)
#     target = torch.from_numpy(target)
#     s1 = s1.to("cuda")
#     s2 = s2.to("cuda")
#     target = target.to("cuda")
#     logits = net(s1, s2)
#     loss = F.cross_entropy(logits, target)
#     if train:
#         loss.backward()
#         optimizer.step()
#     max_index = logits.max(dim=1)[1]
#     mask_correct = max_index == target
#     cnt_correct = mask_correct.sum()
#     return float(loss), int(cnt_correct)


# def train_epoch(net, optimizer, iter, train=True):
#     net.to("cuda")
#     losses = []
#     cnt_correct = 0
#     for i in range(len(iter.batches)):
#         loss, correct_batch = train_batch(net, optimizer, iter.batches[i], train)
#         losses.append(loss)
#         cnt_correct += correct_batch
#     return np.mean(losses), cnt_correct / iter.cnt_samples


# def main():
#     scheduler = None
#     if len(sys.argv) < 2:
#         print("run main.py config.yaml")
#         return
#     path_config = sys.argv[1]
#     with open(path_config, "r") as cfg:
#         params = yaml.load(cfg, Loader=yaml.SafeLoader)
#     path_results_base = "./out/NLI"
#     params["path_results"] = get_unique_results_path(path_results_base)
#     save_data_json(params, os.path.join(params["path_results"], "metadata.json"))
#     embs = vecto.embeddings.load_from_dir(params["path_embeddings"])
#     print("loaded embeddings")
#     net = Net(embs)
#     print("constructed a model")
#     batch_size = 4
#     train_tuples = read_ds(params["path_train"], embs, params["test"])
#     val_tuples = read_ds(params["path_val"], embs, params["test"])
#     it_train = Iterator(train_tuples, batch_size)
#     it_val = Iterator(val_tuples, batch_size)
#     optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad == True], lr=0.001)
#     params["train_log"] = []
#     print("start training")
#     params["time_start_training"] = timer()
#     for id_epoch in range(params["cnt_epochs"]):
#         loss, acc = train_epoch(net, optimizer, it_train)
#         loss_val, acc_val = train_epoch(net, optimizer, it_val, False)
#         time_end = timer()
#         time_total = (time_end - params["time_start_training"])
#         epoch_stats = {}
#         epoch_stats["loss"] = loss
#         epoch_stats["accuracy"] = acc
#         epoch_stats["val_loss"] = loss_val
#         epoch_stats["val_accuracy"] = acc_val
#         params["train_log"].append(epoch_stats)
#         make_snapshot(net, optimizer, scheduler, id_epoch, params)
#         time_end_snap = timer()
#         print(id_epoch,
#               f"loss: {loss:.4f}",
#               f"acc: {acc:.4f}",
#               f"loss_val: {loss_val:.4f}",
#               f"acc_val: {acc_val:.4f}",
#               f"time total: {datetime.timedelta(seconds=time_total)}",
#               f"time snap: {int(time_end_snap - time_end)}s")


if __name__ == "__main__":
    main()
