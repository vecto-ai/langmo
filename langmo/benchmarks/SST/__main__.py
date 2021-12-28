import torch
import torch.nn.functional as F
from langmo.benchmarks.base import (BaseClassificationModel,
                                    ClassificationFinetuner,
                                    aggregate_batch_stats)
from protonn.distributed import dist_adapter as da

from .data import SSTDataModule


class ClassificationModel(BaseClassificationModel):
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, targets)
        mask_correct = torch.argmax(logits, axis=1) == targets
        cnt_correct = mask_correct.sum()
        metrics = {
            f"val_loss": loss,
            f"cnt_correct": cnt_correct,
            f"cnt_questions": torch.tensor(targets.shape[0]),
        }
        return metrics

    def validation_epoch_end(self, outputs):
        print("### validation epoch end")
        metrics = self.hparams["train_logs"][-1]
        # self.add_epoch_id_to_metrics(metrics)
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        loss = da.allreduce(loss)
        cnt_correct = aggregate_batch_stats(outputs, "cnt_correct")
        cnt_questions = aggregate_batch_stats(outputs, "cnt_questions")
        metrics["val_acc"] = (cnt_correct / cnt_questions).item()
        metrics["val_loss"] = loss.item()
        # TODO: make sure metrics are logged
        # self.save_metrics_and_model(metrics)


def main():
    name_task = "SST"
    finetuner = ClassificationFinetuner(name_task, SSTDataModule, ClassificationModel)
    finetuner.run()


if __name__ == "__main__":
    main()
