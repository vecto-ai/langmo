import logging
import os
import sys

from vecto.benchmarks.analogy import Benchmark as Analogy
from vecto.benchmarks.analogy import get_mean_accuracy, get_mean_reciprocal_rank
from vecto.data import get_dataset_by_name
from vecto.embeddings import load_from_dir
from vecto.utils.data import jsonify, save_json

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def report_neigbours(embs, path_dest, seeds=None):
    if seeds is None:
        seeds = ["man", "woman", "quick", "fast", "one", "red"]
    results = []
    for w in seeds:
        neigbours = embs.get_most_similar_words(w)
        results.append([w, neigbours])
    save_json(jsonify(results), os.path.join(path_dest, "results.json"))


def run(path_emb, path_dest):
    # name_dataset = "dummy_analogy"
    name_dataset = "BATS"
    embeddings = load_from_dir(path_emb)
    embeddings.cache_normalized_copy()
    report_neigbours(embeddings, os.path.join(path_dest, "neigbours"))

    # --- run analogy --------------
    # TODO: iterate over supported benchmarks
    # return
    ds = get_dataset_by_name(name_dataset)
    bench_analogy = Analogy(method="LRCos")
    results = bench_analogy.run(embeddings, ds)
    path_dest = os.path.join(path_dest, "analogy")
    summary = {}
    summary["mean_reciprocal_rank"] = get_mean_reciprocal_rank(results)
    summary["mean_accuracy"] = get_mean_accuracy(results)
    save_json(jsonify(results), os.path.join(path_dest, "results_detailed.json"))
    save_json(jsonify(summary), os.path.join(path_dest, "results.json"))
    print("done")


def main():
    path_emb = sys.argv[1]
    if len(sys.argv) == 3:
        path_dest = sys.argv[2]
    else:
        path_dest = os.path.join(path_emb, "eval")
    run(path_emb, path_dest)


if __name__ == "__main__":
    main()
