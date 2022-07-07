from functools import reduce
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn, tanh
from torchmetrics import SQuAD, MeanMetric
from transformers import AutoModelForQuestionAnswering, PretrainedConfig, AutoConfig

from langmo.benchmarks.base import BaseClassificationModel


class QAModel(BaseClassificationModel):
    def __init__(self, net=None, tokenizer=None, params=None):
        super().__init__(net, tokenizer, params)

        self.n_best = params["n_best"]
        self.max_answer_length = params["max_answer_length"]
        self.squad_metric = SQuAD()
        self.val_loss = MeanMetric()
        self.example_to_features = {}
        self.sample_count = 0

    def forward(self, inputs):
        return self.net(**inputs)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch[0]
        outs = self(inputs)
        loss = outs["loss"]
        metrics = {"train_loss": loss.clone().detach().item()}
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, other_features = batch
        outs = self(inputs)

        for example_id in other_features["example_id"]:
            if example_id in self.example_to_features:
                self.example_to_features[example_id].append(self.sample_count)
            else:
                self.example_to_features[example_id] = [self.sample_count]
            self.sample_count += 1

        self.val_loss.update(outs["loss"])
        # here we retain many infos from all batches for validation_epoch_end
        return {
            "start_logits": outs["start_logits"].detach(),
            "end_logits": outs["end_logits"].detach(),
            "offset_mapping": other_features["offset_mapping"],
            "context": other_features["context"],
            "answers": other_features["answers"],
            "has_ans_logits": outs["has_ans_logits"].detach(),
        }

    def validation_epoch_end(self, validation_step_outputs):
        start_logits = gather_outputs(validation_step_outputs, "start_logits")
        end_logits = gather_outputs(validation_step_outputs, "end_logits")
        has_ans_logits = gather_outputs(validation_step_outputs, "has_ans_logits")
        offset_mapping = gather_outputs(validation_step_outputs, "offset_mapping")
        contexts = gather_outputs(validation_step_outputs, "context")
        answers = gather_outputs(validation_step_outputs, "answers")

        pred_answers = []
        true_answers = []
        for example_id, feature_idxs in self.example_to_features.items():
            # get the real answer
            true_answers.append({"answers": answers[feature_idxs[0]], "id": example_id})
            answer_pool = []
            for feature_index in feature_idxs:

                if np.argmax(has_ans_logits[feature_index]) == 0:
                    logit_score = start_logits[feature_index][0] + end_logits[feature_index][0]
                    candidate_answer = {
                        "id": example_id,
                        "prediction_text": "",
                        "logit_score": logit_score,
                    }
                else:
                    candidate_answer = {"id": example_id, "prediction_text": ""}
                    candidate_answer = self._logit_grid_search(
                        start_logits[feature_index],
                        end_logits[feature_index],
                        contexts[feature_index],
                        offset_mapping[feature_index],
                        candidate_answer,
                    )

                candidate_answer["logit_score"] = candidate_answer.get("logit_score", -1.0e-6)
                answer_pool.append(candidate_answer)

            if all(ans["prediction_text"] == "" for ans in answer_pool):
                predicted_answer = {"id": example_id, "prediction_text": ""}
            else:
                predicted_answer = max(answer_pool, key=lambda x: x["logit_score"])

            predicted_answer.pop("logit_score", None)
            pred_answers.append(predicted_answer)

        self.squad_metric.update(pred_answers, true_answers)
        metrics = self.hparams["train_logs"][-1]
        for key, val in self.squad_metric.compute().items():
            metrics[key] = val.item()
        metrics["val_loss"] = self.val_loss.compute().item()
        self.log_dict(metrics)
        self.val_loss.reset()
        self.squad_metric.reset()
        self.example_to_features = {}
        self.sample_count = 0

    def _logit_grid_search(self, start_logit, end_logit, context, offsets, candidate_answer):
        # takes top self.n_best (start/end)_logits
        start_indexes = np.argsort(start_logit)[-1 : -self.n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -self.n_best - 1 : -1].tolist()

        # do a grid search among the top rated pairs using the sum of
        # start_logit and end_logit as score
        for start_index in start_indexes:
            for end_index in end_indexes:
                logit_score = start_logit[start_index] + end_logit[end_index]
                # Skip answers that are not fully in the context
                if not self._check_in_bounds(start_index, end_index, offsets):
                    continue

                if (
                    "logit_score" not in candidate_answer
                    or logit_score - candidate_answer["logit_score"] > 0.0
                ):
                    candidate_answer["prediction_text"] = context[
                        offsets[start_index][0] : offsets[end_index][1]
                    ]
                    candidate_answer["logit_score"] = logit_score

        # logit_zero = start_logit[0] + end_logit[0]
        # if (
        #     "logit_score" not in candidate_answer
        #     or logit_zero - candidate_answer["logit_score"] > 0.0
        # ):
        #     candidate_answer["prediction_text"] = ""
        #     candidate_answer["logit_score"] = logit_zero

        return candidate_answer

    def _check_in_bounds(self, start_index, end_index, offsets):
        if offsets[start_index] is None or offsets[end_index] is None:
            return False
        if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
            return False
        return True


def gather_outputs(outs, key):
    if len(outs) == 0:
        return outs
    if isinstance(outs[0][key], list):
        return reduce(lambda x, y: x + y, [out[key] for out in outs])
    elif isinstance(outs[0][key], torch.Tensor):
        return torch.vstack([out[key] for out in outs]).cpu().numpy()


class QANet(nn.Module):
    def __init__(self, name_model):
        super().__init__()
        if isinstance(name_model, str):
            self.qa = AutoModelForQuestionAnswering.from_pretrained(name_model)
        elif isinstance(name_model, PretrainedConfig):
            self.qa = AutoModelForQuestionAnswering.from_config(name_model)

        self.config = self.qa.config
        self._partial_init(self.config)

    def _partial_init(self, config):
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.cl_loss = nn.CrossEntropyLoss()

    def _classifier_forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def forward(self, **x):
        has_answer = x.pop("has_answer")
        out = self.qa(**x, output_hidden_states=True)
        out["qa_loss"] = out["loss"]
        out["has_ans_logits"] = self._classifier_forward(out["hidden_states"][-1])
        out["has_ans_loss"] = self.cl_loss(out["has_ans_logits"], has_answer)
        out["loss"] = out["qa_loss"] + out["has_ans_loss"]
        return out

    def save_pretrained(self, path):
        p = Path(path)
        self.config.save_pretrained(p)
        torch.save(self.state_dict(), p / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, path):
        p = Path(path)
        if p.exists():
            config = AutoConfig.from_pretrained(p / "config.json")
            with open(p / "config.json") as config_input_file:
                json_conf = json.load(config_input_file)
            config = type(config).from_dict(json_conf)
            n = cls(config)
            n.load_state_dict(torch.load(p / "pytorch_model.bin"))
        else:
            n = cls(path)
        return n
