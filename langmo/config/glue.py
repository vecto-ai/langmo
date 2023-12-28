import yaml

from langmo.config.base import TASKTOMETRIC
from langmo.config.finetune import ConfigFinetune

TASKS_YAML = """
cola:
    keys:
        - sentence
        - null
    cnt_labels: 2
    dataset_prefix: glue
mrpc:
    keys:
        - sentence1
        - sentence2
    cnt_labels: 2
    dataset_prefix: glue
qnli:
    keys:
        - question
        - sentence
    cnt_labels: 2
    dataset_prefix: glue
qqp:
    keys:
        - question1
        - question2
    cnt_labels: 2
rte:
    keys:
        - sentence1
        - sentence2
    cnt_labels: 2
    dataset_prefix: glue
sst2:
    keys:
        - sentence
        - null
    cnt_labels: 2
    dataset_prefix: glue
stsb:
    keys:
        - sentence1
        - sentence2
    cnt_labels: 1
    dataset_prefix: glue
wnli:
    keys:
        - sentence1
        - sentence2
    cnt_labels: 2
    dataset_prefix: glue
mnli:
    keys:
        - premise
        - hypothesis
    cnt_labels: 3
    dataset_prefix: glue
    validation_split:
        - dataset: [glue, mnli]
          split: validation_matched
          name: val
        - dataset: [glue, mnli]
          split: validation_mismatched
          name: val_mismatched
        - dataset: [hans]
          split: validation
          name: hans
NLI:
    keys:
        - premise
        - hypothesis
    cnt_labels: 3
    dataset_prefix: glue
boolq:
    keys:
        - question
        - passage
    cnt_labels: 2
    dataset_prefix: superglue
"""

GLUE_TASKS = yaml.load(TASKS_YAML, Loader=yaml.SafeLoader)


class GLUEConfig(ConfigFinetune):
    def set_defaults(self):
        super().set_defaults()
        self.defaults["metric_name"] = TASKTOMETRIC[self["name_task"]]
        self.defaults["classifier"] = "huggingface"
        # TODO: support custom additional validation splits
        self.defaults["validation_split"] = None

    def _validate(self):
        try:
            assert self["name_task"] in GLUE_TASKS
        except Exception as e:
            glue_tasks = list(GLUE_TASKS.keys())
            raise AssertionError(f"Task must be one of {glue_tasks}") from e

    def _postprocess(self):
        _glue_postprocess(self)


def _glue_postprocess(config):
    # TODO: metric to monitor should be moved here as well
    task_spec = {"validation_split": [{"dataset": ["glue", config["name_task"]], "split": "validation", "name": "val"}]}
    task_spec.update(GLUE_TASKS[config["name_task"]])

    # TODO: moving this bit into this separate _glue_postprocessing function
    # makes it very hard to keep track of in the other bit for when the
    # custom validation split will be implemented
    if config.get("validation_split", None) is None:
        config["validation_split"] = task_spec["validation_split"]

    print("task spec", task_spec)
    config["sent1"] = task_spec["keys"][0]
    config["sent2"] = task_spec["keys"][1]
    config["glue_type"] = task_spec["dataset_prefix"]
    config["num_labels"] = task_spec["cnt_labels"]
    config["validation_split_names"] = [split["name"] for split in task_spec["validation_split"]]
