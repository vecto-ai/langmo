import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import horovod.torch as hvd
torch.manual_seed(0)


class CFDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        # create identical tensors
        cnt_samples = 4
        tensor_x = torch.rand((cnt_samples, 4, 128, 128, 128),
                              out=None,
                              dtype=torch.float32,
                              layout=torch.strided,
                              device=None,
                              requires_grad=False)
        tensor_y = torch.ones((cnt_samples, 4), dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        # one rank should have different one
        # create identical tensors
        cnt_samples = 2
        tensor_x = torch.rand((cnt_samples, 4, 128, 128, 128),
                              out=None,
                              dtype=torch.float32,
                              layout=torch.strided,
                              device=None,
                              requires_grad=False)
        if hvd.rank() == 0:
            tensor_y = torch.zeros((cnt_samples, 4), dtype=torch.float32)
        else:
            tensor_y = torch.ones((cnt_samples, 4), dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=2)
