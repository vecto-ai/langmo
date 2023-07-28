import unittest

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from langmo.training.mlm.data import BatchIter


class MaskTestCase:
    def __init__(self, mask_special_tokens=False):
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        self.params = {
            "batch_size": 16,
            "max_length": 128,
            "cnt_samples_per_epoch": 16,
            "cnt_workers": 1,
            "proba_masking": 0.12,
            "proba_random": 0.015,
            "mask_special_tokens": mask_special_tokens,
        }
        self.batch_iter = BatchIter(
            [],
            self.tokenizer,
            self.params,
        )

    def preprocess_function(self, x, max_length):
        return self.tokenizer(
            x["premise"],
            x["hypothesis"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )

    def adapted_mask_line(self, line, tokenizer, ignore_token_id):
        masked, labels = self.batch_iter.mask_line(line, tokenizer, ignore_token_id)
        return {"masked": masked, "labels": labels}

    def mask(self):
        ds = load_dataset("glue", "mnli", split="train")
        ds = ds.select(range(10000))

        encoded_ds = ds.map(lambda x: self.preprocess_function(x, max_length=128), batched=True)
        encoded_ds = encoded_ds.map(
            lambda x: self.adapted_mask_line(x["input_ids"], self.tokenizer, -100),
            batched=False,
        )
        encoded_ds.set_format(type="pytorch", columns=["input_ids", "masked", "labels"])

        input_tensor = torch.cat(encoded_ds["input_ids"]).reshape(-1)
        masked_tensor = torch.cat(encoded_ds["masked"]).reshape(-1)
        loss_counted_tensor = torch.cat(encoded_ds["labels"])

        mask = torch.ones_like(masked_tensor, dtype=bool)
        global_mask = torch.ones_like(masked_tensor, dtype=bool)
        if not self.params["mask_special_tokens"]:
            for special_tok_id in self.tokenizer.all_special_ids:
                mask = mask & ~(masked_tensor == special_tok_id)
                global_mask = global_mask & ~(input_tensor == special_tok_id)
        else:
            mask = (
                mask
                & ~(masked_tensor == self.tokenizer.pad_token_id)
                & ~(masked_tensor == self.tokenizer.mask_token_id)
            )
            global_mask = global_mask & ~(input_tensor == self.tokenizer.pad_token_id)

        n_masked = (masked_tensor == self.tokenizer.mask_token_id).sum().item()
        n_switched = (input_tensor[mask] != masked_tensor[mask]).sum().item()
        n_loss_counted = (loss_counted_tensor[global_mask] != -100).sum().item()
        n_total = global_mask.sum().item()

        return n_masked / n_total, n_switched / n_total, n_loss_counted / n_total

    # TODO: hack of threading part from batch iter
    # def test_batch(self):
    #     s1 = "how do you do?"
    #     s2 = "I like appricot juice"
    #     batch = [self.tokenizer.tokenize(s1), self.tokenizer.tokenize(s2)]
    #     batch_iter =
    #     encoded =


class Tests(unittest.TestCase):
    def setUp(self):
        self.tester_special_tokens = MaskTestCase(mask_special_tokens=False)
        self.tester_pad_token = MaskTestCase(mask_special_tokens=True)

    @unittest.skip("Temporarily disable")
    def test_mask_special_tokens(self):
        (
            n_masked_ratio,
            n_switched_ratio,
            n_loss_counted_ratio,
        ) = self.tester_special_tokens.mask()
        # test masked tokens
        self.assertAlmostEqual(n_masked_ratio, 0.12, places=2)
        # test random switched tokens
        self.assertAlmostEqual(n_switched_ratio, 0.015, places=3)
        # test that all the 15% of labels are counted in the loss
        self.assertAlmostEqual(n_loss_counted_ratio, 0.15, places=2)

    @unittest.skip("Temporarily disable")
    def test_mask_pad_token(self):
        (
            n_masked_ratio,
            n_switched_ratio,
            n_loss_counted_ratio,
        ) = self.tester_pad_token.mask()
        # test masked tokens
        self.assertAlmostEqual(n_masked_ratio, 0.12, places=2)
        # test random switched tokens
        self.assertAlmostEqual(n_switched_ratio, 0.015, places=3)
        # test that all the 15% of labels are counted in the loss
        self.assertAlmostEqual(n_loss_counted_ratio, 0.15, places=2)


if __name__ == "__main__":
    unittest.main()
