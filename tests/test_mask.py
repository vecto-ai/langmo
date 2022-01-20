import unittest

from langmo.pretraining.data import BatchIter, mask_line
from transformers import AutoTokenizer
from datasets import load_dataset
import torch



class Tests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    
    def preprocess_function(self, x, max_length):
            return self.tokenizer(
                x["premise"],
                x["hypothesis"],
                padding=False,
                truncation=True,
                max_length=max_length
            )

    def adapted_mask_line(self, line, tokenizer, ignore_token_id):
        masked, labels = mask_line(line, tokenizer, ignore_token_id)
        return {"masked":masked, "labels":labels}

    def test_mask(self):
        ds = load_dataset("glue", "mnli", split="train")
        ds = ds.select(range(50000))

        encoded_ds = ds.map(
            lambda x: self.preprocess_function(x, max_length=128), batched=True
        )
        encoded_ds = encoded_ds.map(
            lambda x: self.adapted_mask_line(x["input_ids"], self.tokenizer, -100), batched=False
        )
        encoded_ds.set_format(type="pytorch", columns=["input_ids", "masked", "labels"])

        input_tensor = torch.cat(encoded_ds["input_ids"]).reshape(-1)
        masked_tensor = torch.cat(encoded_ds["masked"]).reshape(-1)
        loss_counted_tensor = torch.cat(encoded_ds["labels"])

        mask = torch.ones_like(masked_tensor, dtype=bool)
        global_mask = torch.ones_like(masked_tensor, dtype=bool)
        for special_tok_id in self.tokenizer.all_special_ids:
            mask = mask & ~(masked_tensor == special_tok_id)
            global_mask = global_mask & ~(input_tensor == special_tok_id)
        
        n_masked = (masked_tensor == self.tokenizer.mask_token_id).sum().item()
        n_switched = (input_tensor[mask] != masked_tensor[mask]).sum().item()
        n_loss_counted = (loss_counted_tensor[global_mask] != -100).sum().item()
        n_total = global_mask.sum().item()

        # test masked tokens
        self.assertAlmostEqual(n_masked/n_total, 0.12, places=2)
        # test random switched tokens
        self.assertAlmostEqual(n_switched/n_total, 0.015, places=3)
        # test that all the 15% of labels are counted in the loss
        self.assertAlmostEqual(n_loss_counted/n_total, 0.15, places=2)

        input_tensor = torch.cat(encoded_ds["input_ids"]).reshape(-1)
        masked_tensor = torch.cat(encoded_ds["masked"]).reshape(-1)
        loss_counted_tensor = torch.cat(encoded_ds["labels"])

        mask = torch.ones_like(masked_tensor, dtype=bool)
        global_mask = torch.ones_like(masked_tensor, dtype=bool)
        for special_tok_id in self.tokenizer.all_special_ids:
            mask = mask & ~(masked_tensor == special_tok_id)
            global_mask = global_mask & ~(input_tensor == special_tok_id)
        
        n_masked = (masked_tensor == self.tokenizer.mask_token_id).sum().item()
        n_switched = (input_tensor[mask] != masked_tensor[mask]).sum().item()
        n_loss_counted = (loss_counted_tensor[global_mask] != -100).sum().item()
        n_total = global_mask.sum().item()

        # test masked tokens
        self.assertAlmostEqual(n_masked/n_total, 0.12, places=2)
        # test random switched tokens
        self.assertAlmostEqual(n_switched/n_total, 0.015, places=3)
        # test that all the 15% of labels are counted in the loss
        self.assertAlmostEqual(n_loss_counted/n_total, 0.15, places=2)
    # TODO: hack of threading part from batch iter
    # def test_batch(self):
    #     s1 = "how do you do?"
    #     s2 = "I like appricot juice"
    #     batch = [self.tokenizer.tokenize(s1), self.tokenizer.tokenize(s2)]
    #     batch_iter =
    #     encoded = 

if __name__ == "__main__":
    unittest.main()