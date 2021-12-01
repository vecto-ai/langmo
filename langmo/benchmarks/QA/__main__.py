"""
Fine-tuning a model on question aswering datasets such as squad
SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 
unanswerable questions written adversarially by crowdworkers to look 
similar to answerable ones.

Reported metrics are 

EM - The Exact Match metric measures the percentage of predictions that match any one of the ground truth answers exactly. 

F1 - = precision * recall / (rpecision + recall), where precision is the ratio of the number of shared words to the total 
number of words in the prediction, and recall  is the ratio of the number of shared words to the total 
number of words in the ground truth

"""

import torch
from protonn.distributed import dist_adapter as da

# import torch.nn.functional as F
from langmo.base import PLBase
from langmo.benchmarks.base import QAFinetuner, aggregate_batch_stats

from .data import QADataModule as DataModule


class QAModel(PLBase):
    def forward(self, inputs):
        return self.net(**inputs)

    def training_step(self, batch, batch_idx):
        inputs, answers = batch[0]
        # 0 is there seince PL returns tuple of batched from all dataloaders
        # not sure if this will be persisten behavior
        outs = self(inputs)
        loss = outs["loss"]
        # acc = accuracy(F.softmax(logits, dim=1), targets)
        metrics = {
            "train_loss": loss,
            # "train_acc": acc,
        }
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, answers = batch
        outs = self(inputs)
        # print(answers)
        # pos_start = torch.nn.functional.softmax(outs["start_logits"])
        pos_start = torch.argmax(outs["start_logits"], axis=1)
        # print("start", pos_start)
        batch_size = len(inputs["input_ids"])
        # print("batch_size", batch_size)
        for i in range(batch_size):
            outs["end_logits"][i, :pos_start[i]] = 0
        pos_end = torch.argmax(outs["end_logits"], axis=1)
        # print("end", pos_end)
        cnt_correct = 0
        for i in range(batch_size):
            predicted_tokens = inputs["input_ids"][i][pos_start[i]:pos_end[i]]
            # print("pred tokens", predicted_tokens)
            str_pred_answer = self.tokenizer.decode(predicted_tokens)
            # print("pred answer", str_pred_answer)
            if str_pred_answer in answers[i]:
                cnt_correct += 1

        # loss = F.cross_entropy(logits, targets)
        # mask_correct = torch.argmax(logits, axis=1) == targets
        # cnt_correct = mask_correct.sum()
        metrics = {
            f"val_loss": outs["loss"],
            f"cnt_correct": torch.tensor(cnt_correct),
            f"cnt_questions": torch.tensor(batch_size),
        }
        return metrics

    def validation_epoch_end(self, outputs):
        # print("### validation epoch end")
        metrics = {}
        self.add_epoch_id_to_metrics(metrics)
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss = da.allreduce(loss)
        cnt_correct = aggregate_batch_stats(outputs, "cnt_correct")
        cnt_questions = aggregate_batch_stats(outputs, "cnt_questions")
        metrics["val_EM"] = cnt_correct / cnt_questions
        metrics["val_loss"] = loss
        self.save_metrics_and_model(metrics)


def main():
    name_task = "QA"
    finetuner = QAFinetuner(name_task, DataModule, QAModel)
    finetuner.run()


if __name__ == "__main__":
    main()
