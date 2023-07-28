# from langmo.nn import Siamese, TopMLP2, wrap_encoder

from pathlib import Path

from protonn.utils import load_json
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

# import langmo
# import transformers
from langmo.nn.classifier import PretrainedClassifier
from langmo.nn.cnet import MLModel

from .classifier import ClassificationHead, Classifier
from .heads import *
from .siamese import *
from .wrappers import *


def is_langmo_model(model_name):
    path_config = Path(model_name) / "config.json"
    if path_config.exists():
        config = load_json(path_config)
        if config["model_type"] == "langmo":
            return True
    return False


def create_mlm(params):
    if is_langmo_model(params["model_name"]):
        net = MLModel.from_pretrained(params["model_name"])
        return net, "langmo"
    else:
        net = AutoModelForMaskedLM.from_pretrained(params["model_name"])
        name_run = params["model_name"]
        return net, name_run


def create_net(params):
    # TODO: allow for custom classification head even if it's not siamese
    # TODO: move creation of model logit to model-related class / submodule
    # TDDO: support models with multiple classification heads
    if is_langmo_model(params["model_name"]):
        net = PretrainedClassifier.from_pretrained(
            params["model_name"], num_labels=params["num_labels"], hidden_size=768
        )
        # TODO: hidden size should be saved to config at pretraining!!!
        return net, "langmo"

    if params["siamese"]:
        name_run = "siam_" + params["encoder_wrapper"] + "_"
        if params["freeze_encoder"]:
            name_run += "fr_"
        name_run += params["model_name"]
        encoder = AutoModel.from_pretrained(params["model_name"], add_pooling_layer=False)
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
                params["model_name"], num_labels=params["num_labels"]
            )
            name_run = params["model_name"].split("pretrain")[-1]
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
