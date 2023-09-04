# import json
import sys

from transformers import pipeline
from vecto.benchmarks.base import Benchmark as BaseBenchmark
from vecto.utils.data import print_json

from langmo.nn import create_mlm


# TODO: we don't know which is mask token
def get_queries(mask_token):
    queries = [
        f"I like {mask_token} beer.",
        f"I like cold {mask_token}.",
        f"Cows produce {mask_token}.",
    ]
    return queries


def fill_mask(model, tokenizer, query):
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
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
        mask_token = self.tokenizer.mask_token
        for query in get_queries(mask_token):
            predicted = fill_mask(self.model, self.tokenizer, query)
            result["results"].append(predicted)
        return result


def main():
    from transformers import AutoTokenizer

    params = {}
    params["model_name"] = sys.argv[1]
    model, name_run = create_mlm(params)
    print("running", name_run)
    model.eval()
    from pathlib import Path

    import torch

    data = torch.load(Path(params["model_name"]) / "pytorch_model.bin")
    print("decoder", data["lm_head.decoder.weight"][0][:10])
    print("encoder", data["encoder.embeddings.weight"][0][:10])
    tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
    benchmark = Benchmark(model, tokenizer)
    result = benchmark.run()
    print_json(result)
