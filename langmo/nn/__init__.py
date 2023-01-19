# from langmo.nn import Siamese, TopMLP2, wrap_encoder

from pathlib import Path

import langmo
import transformers
from langmo.nn.classifier import PretrainedClassifier
from protonn.utils import load_json
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from .classifier import ClassificationHead, Classifier
from .cnet import Encoder
from .heads import *
from .siamese import *
from .wrappers import *


def create_net(params):
    # TODO: allow for custom classification head even if it's not siamese
    # TODO: move creation of model logit to model-related class / submodule
    # TDDO: support models with multiple classification heads
    CONFIG_MAPPING_NAMES["langmo"] = "langmo"
    transformers.models.langmo = langmo.nn.cnet
    name_model = params["model_name"]
    # TODO: try if path, load config
    # if it's our model - do custom load, else rely on HF
    if (Path(params["model_name"]) / "config.json").is_file:
        config = load_json(Path(params["model_name"]) / "config.json")
        if config["model_type"] == "langmo":
            # encoder = Encoder()
            # head = ClassificationHead(num_labels=params["num_labels"])
            # return Classifier(encoder, head), "langmo"
            net = PretrainedClassifier.from_pretrained(params["model_name"])
            return net, "langmo"

    if params["siamese"]:
        name_run = "siam_" + params["encoder_wrapper"] + "_"
        if params["freeze_encoder"]:
            name_run += "fr_"
        name_run += name_model
        encoder = AutoModel.from_pretrained(name_model, add_pooling_layer=False)
        wrapped_encoder = wrap_encoder(
            encoder, name=params["encoder_wrapper"], freeze=params["freeze_encoder"]
        )
        # TODO: add different heads support
        net = Siamese(
            wrapped_encoder,
            # *4 logic should be inside siamese and not here
            TopMLP2(
                in_size=wrapped_encoder.get_output_size() * 4,
                cnt_classes=params["num_labels"],
            ),
        )
    else:
        if params["classifier"] == "huggingface":
            net = AutoModelForSequenceClassification.from_pretrained(
                name_model, num_labels=params["num_labels"]
            )
            name_run = name_model.split("pretrain")[-1]
        # TODO: decide what classifier param ataully should be
        elif params["encoder_wrapper"] == "lstm":
            encoder = AutoModel.from_pretrained(name_model, add_pooling_layer=False)
            wrapped_encoder = wrap_encoder(
                encoder, name=params["encoder_wrapper"], freeze=params["freeze_encoder"]
            )
            head = TopMLP2(
                in_size=wrapped_encoder.get_output_size(),
                cnt_classes=params["num_labels"],
            )
            net = Classifier(wrapped_encoder, head)
            name_run = name_model + "_custom_cls"
        else:
            raise NotImplemented("not yet")

    return net, name_run
