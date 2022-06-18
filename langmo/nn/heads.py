import torch.nn as nn
import torch.nn.functional as F


class TopMLP2(nn.Module):
    def __init__(self, in_size=512, hidden_size=512, cnt_classes=3):
        super().__init__()
        self.l1 = nn.Linear(in_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, cnt_classes)

    def forward(self, x):
        h = self.l1(x)
        h = F.relu(h)
        h = self.l2(h)
        return h


# def get_downstream_head(name):
#     name = name.lower()
#     wrapper = {"topmlp2":TopMLP2, "lstm":"LSTM_Encoder"}
#     return wrapper[name]
