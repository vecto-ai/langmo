import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, encoder, head, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.config = encoder.net.config

    def forward(self, **inputs):
        h = self.encoder(**inputs)
        h = self.head(h)
        return {"logits": h}

    def save_pretrained(self, path):
        # TODO: do proper warning
        # TODO: save encoder part
        print("save poretrained not implemented")

    # TODO: make issue in torch for not supporting properties
    # @property
    # def config(self):
    #     return self.encoder.config
