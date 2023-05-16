import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


# TODO: use proper names
# TODO: distinguish lstm from other encoders
class BaseConfig(PretrainedConfig):
    vocab_size = 50265
    hidden_size = 768
    initializer_range = 0.02
    cnt_layers = 8
    dropout = 0.2
    model_type = "langmo"


class BaseCNet(PreTrainedModel):
    config_class = BaseConfig


class RNNLayer(nn.Module):
    def __init__(self,
                 nhid,
                 dropout):
        super().__init__()
        cnt_heads = 6
        hidden_size = 128
        self.rnn = nn.ModuleList([nn.LSTM(input_size=nhid,
                                          hidden_size=hidden_size,
                                          num_layers=1,
                                          dropout=dropout,
                                          bidirectional=True,
                                          batch_first=True) for _ in range(cnt_heads)])
        self.intermediate = nn.Linear(hidden_size * 2 * cnt_heads, nhid * 4)
        self.output = nn.Linear(nhid * 4, nhid)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, embeddings):
        outs = [h(embeddings)[0] for h in self.rnn]
        concatenated = torch.cat(outs, dim=2)
        intermediate = nn.functional.gelu(self.intermediate(concatenated))
        projected = self.output(intermediate)
        dropouted = self.dropout(projected)
        # TODO: weighted residual connection?
        residual = embeddings + dropouted
        normed = self.layer_norm(residual)
        # TODO: what is apply_chunking_to_forward() in HF roberta
        return normed


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO move embedding dropout and norming into Embedding layer
        self.drop = nn.Dropout(config.dropout)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # TODO: this seemed to actually perform a bit better though slower
        # refactor it as one of the options for the layer
        # self.rnn = nn.LSTM(config.hidden_size,
        #                    config.hidden_size,
        #                    config.cnt_layers,
        #                    dropout=config.dropout,
        #                    bidirectional=(config.cnt_directions == 2),
        #                    batch_first=True)
        self.layers = nn.Sequential(*[RNNLayer(config.hidden_size, config.dropout) for _ in range(config.cnt_layers)])

        # self.init_weights(embs)

    # def init_weights(self, embs):
        # initrange = 0.1
        # self.embs.weight.data = torch.from_numpy(embs.matrix)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids):
        emb = self.drop(self.embeddings(input_ids))
        # output, self.hidden = self.rnn(emb, self.hidden)
        output = self.layers(emb)
        return output


class LMHead(nn.Module):

    def __init__(self, config, eps=0.00001):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # TODO: do we really have to do this?
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
        # TODO: here we rely on torch x-entropy ignoring label -100
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            loss = None
        return {"logits": logits, "loss": loss}

    # def get_input_embeddings(self):
    #     return self.encoder.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.encoder.embeddings.word_embeddings = value

    # TODO: init weights
    # init weights


def get_mlmodel(params):
    config = BaseConfig()
    # TODO: tie output and input embeddings properly (now in init)
    return MLModel(config)
