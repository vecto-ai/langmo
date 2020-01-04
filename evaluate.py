import numpy as np
from vecto.embeddings import load_from_dir
from vecto.benchmarks.analogy import Analogy
from vecto.data import get_dataset_by_name


# TODO: take this from config or command line args
# TODO: and dataset name or path

path_emb = "/home/blackbird/Cloud/remote/berlin/data/NLP/embeddings/test/w2v_ref/brown_thrd24__SG_d128_w3_neg4_i10"


# TODO: move this to vecto result parsing
def get_mean_reciprocal_rank(results):
    mean_reciprocal_rank=np.mean([(lambda r : 0 if r<=0 else 1/r) (experiment["rank"]) for category in results for experiment in category["details"] ])
    return mean_reciprocal_rank

def get_mean_accuracy(results):
    mean_accuracy=np.mean([experiment["rank"]==0 for category in results for experiment in category["details"] ])
    return mean_accuracy


embeddings = load_from_dir(path_emb)
embeddings.cache_normalized_copy()
# ds = get_dataset_by_name("BATS")
ds = get_dataset_by_name("dummy_analogy")
bench_analogy = Analogy()
results = bench_analogy.run(embeddings, ds)

summary = {}
summary["mean_reciprocal_rank"] = get_mean_reciprocal_rank(results)
summary["mean_accuracy"] = get_mean_accuracy(results)

print(summary)
# save results to target dir
