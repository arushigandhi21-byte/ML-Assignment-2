import json
import sys

import numpy as np

from common.config import RESULTS_DIR, LOG_DIR


class Tee:

    def __init__(self, name):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.f = open(LOG_DIR / f"{name}.log", "w")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def _to_builtin(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=_to_builtin)


def load_json(name):
    with open(RESULTS_DIR / f"{name}.json") as f:
        return json.load(f)


def leaf_stats(tree):
    t = tree.tree_
    is_leaf = t.children_left == -1
    samples = t.n_node_samples[is_leaf]
    return {
        "n_leaves": int(is_leaf.sum()),
        "mean_samples_per_leaf": float(samples.mean()),
        "frac_single_sample_leaves": float((samples == 1).mean()),
        "depth": int(tree.get_depth()),
    }
