import os
import sys
import logging
from vecto.embeddings import load_from_dir
from vecto.benchmarks.analogy import Benchmark as Analogy
from vecto.benchmarks.analogy import get_mean_accuracy, get_mean_reciprocal_rank
from vecto.data import get_dataset_by_name
from vecto.utils.data import save_json, jsonify

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def run(path_emb, path_dest):
    # name_dataset = "dummy_analogy"
    name_dataset = "BATS"
    embeddings = load_from_dir(path_emb)
    embeddings.cache_normalized_copy()
    ds = get_dataset_by_name(name_dataset)
    bench_analogy = Analogy(method="LRCos")
    results = bench_analogy.run(embeddings, ds)

    summary = {}
    summary["mean_reciprocal_rank"] = get_mean_reciprocal_rank(results)
    summary["mean_accuracy"] = get_mean_accuracy(results)
    save_json(jsonify(results), os.path.join(path_dest, "results_detailed.json"))
    save_json(jsonify(summary), os.path.join(path_dest, "results.json"))
    print("done")


def main():
    path_emb = sys.argv[1]
    if len(sys.argv == 3):
        path_dest = sys.argv[2]
    else:
        path_dest = os.path.join(path_emb, "/eval/analogy")
    run(path_emb, path_dest)


if __name__ == "__main__":
    main()
