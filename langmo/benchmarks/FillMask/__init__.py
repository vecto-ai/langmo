import json
import sys

from transformers import pipeline
from vecto.benchmarks.base import Benchmark as BaseBenchmark
from vecto.utils.data import print_json


# TODO: we don't know which is mask token
def get_queries(mask_token):
    queries = [
        f"I like {mask_token} beer.",
        f"I like cold {mask_token}.",
        f"Cows produce {mask_token}.",
    ]
    return queries


def fill_mask(model, tokenizer, query):
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer)
    result = fill_mask(query)
    return result


class Benchmark(BaseBenchmark):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def run(self):
        # TODO: reuse this from some parent class code
        # TODO: move specific metadata to separate method anywya
        result = dict()
        result["setup"] = dict()
        result["setup"]["name"] = "fill_mask"
        result["setup"]["desciption"] = "Fill masked tokens"
        result["results"] = []
        # TODO: get this from model config
        for query in get_queries("[MASK]"):
            predicted = fill_mask(self.model, self.tokenizer, query)
            result["results"].append(predicted)
        return result


def main():
    from transformers import AutoModelForMaskedLM
    from transformers import AutoTokenizer
    name_model = sys.argv[1]
    model = AutoModelForMaskedLM.from_pretrained(name_model)
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    benchmark = Benchmark(model, tokenizer)
    result = benchmark.run()
    print_json(result)
