from vecto.embeddings import load_from_dir
from vecto.benchmarks.analogy import Benchmark as Analogy
from vecto.benchmarks.analogy import get_mean_accuracy, get_mean_reciprocal_rank
from vecto.data import get_dataset_by_name
from vecto.utils.data import save_json, jsonify
import logging
# TODO: take this from config or command line args
# TODO: and dataset name or path

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

path_emb = "/home/blackbird/Cloud/remote/berlin/data/NLP/embeddings/test/w2v_ref/brown_thrd24__SG_d128_w3_neg4_i10"
name_dataset = "dummy_analogy"  # "BATS"
path_target = "/tmp/temp_target"


embeddings = load_from_dir(path_emb)
embeddings.cache_normalized_copy()
ds = get_dataset_by_name(name_dataset)
bench_analogy = Analogy()
results = bench_analogy.run(embeddings, ds)

save_json(jsonify(results), "/tmp/vecto/results.json")

# TODO: move summarization of results to a separate module
summary = {}
summary["mean_reciprocal_rank"] = get_mean_reciprocal_rank(results)
summary["mean_accuracy"] = get_mean_accuracy(results)

print(summary)
# save_json()
# save results to target dir
