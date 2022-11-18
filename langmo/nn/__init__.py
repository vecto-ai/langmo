# from langmo.nn import Siamese, TopMLP2, wrap_encoder

from transformers import AutoModel, AutoModelForSequenceClassification

from .heads import *
from .siamese import *
from .wrappers import *


def create_net(params):
    # TODO: allow for custom classification head even if it's not siamese
    # TODO: move creation of model logit to model-related class / submodule
    # TDDO: support models with multiple classification heads
    name_model = params["model_name"]
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
            TopMLP2(
                in_size=wrapped_encoder.get_output_size() * 4,
                cnt_classes=params["num_labels"],
            ),
        )
    else:
        net = AutoModelForSequenceClassification.from_pretrained(
            name_model, num_labels=params["num_labels"]
        )
        name_run = name_model.split("pretrain")[-1]

    return net, name_run
