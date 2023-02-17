import torch
import torch.nn as nn
from langmo.nn.cnet import BaseCNet, BaseConfig, Encoder


class ClassificationHead(nn.Module):
    """Roblerta-like head for sentence-level classification tasks."""
    # TODO: use config here
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PretrainedClassifier(BaseCNet):
    def __init__(self, config, **kwargs):
        _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
        print("INIT Classifier with ", kwargs)
        super().__init__(config)
        self.config = config
        self.encoder = Encoder(config)
        self.classifier = ClassificationHead(hidden_size=self.config.hidden_size * 2,
                                             num_labels=self.config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        encoded = self.encoder(input_ids)
        logits = self.classifier(encoded)
        return {"logits": logits}

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Classifier(BaseCNet):
    def __init__(self, encoder, head):
        super().__init__(config)
        self.encoder = encoder
        self.classifier = head

    def forward(self, input_ids, attention_mask=None):
        encoded = self.encoder(input_ids)
        logits = self.classifier(encoded)
        return {"logits": logits}


# class Classifier(nn.Module):
#     def __init__(self, encoder, head, freeze_encoder=True):
#         super().__init__()
#         self.encoder = encoder
#         self.head = head
#         self.config = encoder.net.config

#     def forward(self, **inputs):
#         h = self.encoder(**inputs)
#         h = self.head(h)
#         return {"logits": h}

#     def save_pretrained(self, path):
#         # TODO: do proper warning
#         # TODO: save encoder part
#         print("save poretrained not implemented")

    # TODO: make issue in torch for not supporting properties
    # @property
    # def config(self):
    #     return self.encoder.config
