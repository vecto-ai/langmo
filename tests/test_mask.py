import unittest

from langmo.pretraining.data import BatchIter, mask_line
from transformers import AutoTokenizer


class Tests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")

    def test_mask(self):
        sentence = "hey, how are you ?"
        tokens = self.tokenizer.tokenize(sentence)
        print("tokens")
        print(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        print("ids")
        print("      ", input_ids)
        masked_ids, labels = mask_line(input_ids, tokenizer)
        print("masked")
        print(masked_ids)
        print("labels")
        print(labels)

    # TODO: hack of threading part from batch iter
    # def test_batch(self):
    #     s1 = "how do you do?"
    #     s2 = "I like appricot juice"
    #     batch = [self.tokenizer.tokenize(s1), self.tokenizer.tokenize(s2)]
    #     batch_iter =
    #     encoded = 
