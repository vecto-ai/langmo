import yaml
from langmo.config.base import TASKTOMETRIC, ConfigFinetune

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

        try:
            assert self["name_task"] in GLUE_TASKS
        except Exception:
            glue_tasks = list(GLUE_TASKS.keys())
            raise AssertionError(f"Task must be one of {glue_tasks}")
        # TODO: metric to monitor should be moved here as well
        task_spec = {"validation_split": [{"dataset": ["glue", self["name_task"]],
                                           "split": "validation",
                                           "name": "val"}]}
        task_spec.update(GLUE_TASKS[self["name_task"]])
        print("task spec", task_spec)
        # TODO: THIS is not really defaults... should be treated differently
        self.defaults["sent1"] = task_spec["keys"][0]
        self.defaults["sent2"] = task_spec["keys"][1]
        self.defaults["num_labels"] = task_spec["cnt_labels"]
        self.defaults["metric_name"] = TASKTOMETRIC[self["name_task"]]
        self.defaults["glue_type"] = task_spec["dataset_prefix"]
        self.defaults["validation_split"] = task_spec["validation_split"]
        self["validation_split_names"] = [split["name"] for split in task_spec["validation_split"]]
