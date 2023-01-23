import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


# TODO: use preoper names
# TODO: distibguish lstm from other encoders
class BaseConfig(PretrainedConfig):
    vocab_size = 50265
    hidden_size = 512
    initializer_range = 0.02
    cnt_layers = 6
    dropout = 0.2
    cnt_directions = 2
    model_type = "langmo"


class BaseCNet(PreTrainedModel):
    config_class = BaseConfig


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.drop = nn.Dropout(config.dropout)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, config.cnt_layers,
                           dropout=config.dropout,
                           bidirectional=(config.cnt_directions == 2),
                           batch_first=True)
        # self.decoder = nn.Linear(nhid, 2)
        # self.init_weights(embs)
        # self.embs.weight.requires_grad = False

    # def init_weights(self, embs):
        # initrange = 0.1
        # self.embs.weight.data = torch.from_numpy(embs.matrix)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids):
        # print("in", input.shape)
        emb = self.drop(self.embeddings(input_ids))
        # emb.unsqueeze_(0)
        # print("emb", emb.shape)
        # output, self.hidden = self.rnn(emb, self.hidden)
        output, _hidden = self.rnn(emb)
        # print("rnn out", output.shape)
        # output = self.drop(output)
        # decoded = self.decoder(output[-1])
        # print("decoded", decoded)
        return output


class LMHead(nn.Module):

    def __init__(self, config, eps=0.00001):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * config.cnt_directions, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # TODO: do we really gave to do this?
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
        self.vocab_size = config.vocab_size

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias
        # TODO: Why not in and out embeddings are tied here???


class MLModel(BaseCNet):
    def __init__(self, config):
        # TODO: fix this later for saving/loading
        # _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
        # _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
        # _keys_to_ignore_on_load_unexpected = [r"pooler"]
        super().__init__(config=config)
        encoder = Encoder(config)
        head = LMHead(config)
        head.decoder.weight = encoder.embeddings.weight
        self.config = config
        self.encoder = encoder
        self.lm_head = head
        self.loss_fct = torch.nn.CrossEntropyLoss()
        # TODO: tie weights????

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded = self.encoder(input_ids)
        logits = self.lm_head(encoded)
        # torch why u not just use last dimensions and treat all preceeding as batch :-\
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            loss = None
        return {"logits": logits, "loss": loss}

    # def get_input_embeddings(self):
    #     return self.encoder.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.encoder.embeddings.word_embeddings = value


def get_mlmodel(params):
    config = BaseConfig()
    # TODO: tie output and input embeddings
    return MLModel(config)
