import numpy as np
from vecto.embeddings import load_from_dir
from vecto.benchmarks.analogy import Analogy
from vecto.data import get_dataset_by_name
from vecto.utils.data import save_json

# TODO: take this from config or command line args
# TODO: and dataset name or path

path_emb = "/home/blackbird/Cloud/remote/berlin/data/NLP/embeddings/test/w2v_ref/brown_thrd24__SG_d128_w3_neg4_i10"
name_dataset = "dummy_analogy"  # "BATS"
path_target = "/tmp/temp_target"

# TODO: move this to vecto result parsing
def get_mean_reciprocal_rank(results):
    mean_reciprocal_rank=np.mean([(lambda r : 0 if r<=0 else 1/r) (experiment["rank"]) for category in results for experiment in category["details"] ])
    return mean_reciprocal_rank

def get_mean_accuracy(results):
    mean_accuracy=np.mean([experiment["rank"]==0 for category in results for experiment in category["details"] ])
    return mean_accuracy


embeddings = load_from_dir(path_emb)
embeddings.cache_normalized_copy()
ds = get_dataset_by_name(name_dataset)
bench_analogy = Analogy()
results = bench_analogy.run(embeddings, ds)

# TODO: move summarization of results to a separate module
summary = {}
summary["mean_reciprocal_rank"] = get_mean_reciprocal_rank(results)
summary["mean_accuracy"] = get_mean_accuracy(results)

print(summary)
# save_json()
# save results to target dir
