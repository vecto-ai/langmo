import torch
from datasets import ReadInstruction, load_dataset
from torch.utils.data import DataLoader, DistributedSampler

from langmo.training.base_data import BaseCollator, BaseDataModule


class Collator(BaseCollator):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.inputkeys = [
            "input_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
            "has_answer",
        ]

    def __call__(self, batch):
        input_features = {}
        other_features = {}
        for sample in batch:
            for key, val in sample.items():
                if key not in self.inputkeys:
                    other_features[key] = other_features.get(key, [])
                    other_features[key].append(val)
                else:
                    input_features[key] = input_features.get(key, [])
                    input_features[key].append(val)

        return {i: torch.tensor(j) for i, j in input_features.items()}, other_features


class QADataModule(BaseDataModule):
    def __init__(self, cluster_env, tokenizer, params):
        super().__init__(cluster_env, tokenizer, params)
        self.collator = Collator(self.tokenizer, params)

        self.max_length = params["max_length"]
        self.stride = params["stride"]
        self.n_best = params["n_best"]

    def setup(self, stage=None):
        # self.cnt_train_samples is set by self.train_dataloader
        self._langmo_train_dataloader = self.get_split_dataloader(self.params["name_task"], split="train")
        self.cnt_train_samples = len(self._langmo_train_dataloader.dataset)

    # pylint: disable=access-member-before-definition
    def train_dataloader(self):
        return [self._langmo_train_dataloader]

    # pylint: disable=access-member-before-definition
    def val_dataloader(self):
        return [self.get_split_dataloader(self.params["name_task"], split="validation")]

    def get_split_dataloader(self, dataset_name, split):
        if self.test:
            read_instruction = ReadInstruction(
                split,
                from_=0,
                to=self.params["batch_size"] * 3,
                unit="abs",
            )
        else:
            if split == "validation":
                # for val we shard DS first so that parts of one question are on one worker
                percent_start = float(self.trainer.global_rank) / float(self.trainer.world_size) * 100
                percent_end = float(self.trainer.global_rank + 1) / float(self.trainer.world_size) * 100
                read_instruction = ReadInstruction(
                    split,
                    from_=percent_start,
                    to=percent_end,
                    unit="%",
                )
            else:
                # for train we have to load whole thing, the preprocess then shart
                # so that each worker has same number of sampler roughly
                read_instruction = split
        ds = load_dataset(dataset_name, split=read_instruction)
        if split == "validation":
            ds = self._map(ds, self.preprocess_validation_examples)
            sampler = None
        elif split == "train":
            ds = self._map(ds, self.preprocess_training_examples)
            sampler = DistributedSampler(ds, self.trainer.world_size, self.trainer.global_rank, self.shuffle)

        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            sampler=sampler,
        )

    def _map(self, ds, func):
        return ds.map(
            func,
            batched=True,
            remove_columns=[i for i in ds.column_names if i != "answers"],
            load_from_cache_file=False,
        )

    def preprocess_validation_examples(self, examples):
        inputs = self.preprocess_training_examples(examples)

        for i in range(len(inputs["input_ids"])):

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

        return inputs

    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        # question then context ordering is done according to HF example, not sure if it's the best :-\
        # https://huggingface.co/docs/transformers/tasks/question_answering
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        # offset_mapping is used to get a sense of strides and to map proper
        # start and stop back into the original sentence in the case
        offset_mapping = inputs.get("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        contexts = examples["context"]
        start_positions = []
        end_positions = []
        new_answers = []
        new_contexts = []
        example_ids = []
        has_ans = []
        for i in range(len(inputs["input_ids"])):
            offset = offset_mapping[i]
            sample_idx = sample_map[i]
            # the idea is that the long context gets split into multiple sequences of max length
            answer = answers[sample_idx]
            context = contexts[sample_idx]
            new_contexts.append(context)
            example_ids.append(examples["id"][sample_idx])
            if len(answer["answer_start"]) == 0 and len(answer["text"]) == 0:
                answer["text"].append("")
                answer["answer_start"].append(0)
                start_pos, end_pos = 0, 0
                has_ans.append(0)
            else:
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                start_pos, end_pos = self._get_bounds(sequence_ids, offset, start_char, end_char)
                has_ans.append(1)

            start_positions.append(start_pos)
            end_positions.append(end_pos)

            new_answers.append(answer)

        inputs["answers"] = new_answers
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["example_id"] = example_ids
        inputs["context"] = new_contexts
        inputs["has_answer"] = has_ans
        return inputs

    def _get_bounds(self, sequence_ids, offset, start_char, end_char):
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            return (0, 0)
        else:
            # Otherwise it's the start and end token positions
            start_idx = context_start
            while start_idx <= context_end and offset[start_idx][0] <= start_char:
                start_idx += 1
            end_idx = context_end
            while end_idx >= context_start and offset[end_idx][1] >= end_char:
                end_idx -= 1
            return (start_idx - 1, end_idx + 1)
