import json

import numpy as np

from common.config import ROOT, SEED, ROLL_NUMBER, SPLIT_PATH, MASK_ROWS, MASK_COLS
from common.utils import load_json

q1a, q1bc, q2b, q3 = load_json("q1a"), load_json("q1bc"), load_json("q2b"), load_json("q3_struct")
split = np.load(SPLIT_PATH)
cfg = {
    "roll_number": ROLL_NUMBER,
    "seed": SEED,
    "split": {"file": "results/split.npz", "per_identity": {"train": 6, "val": 2, "test": 2},
              "sizes": {k: int(len(split[k])) for k in ("train", "val", "test")}},
    "features": "raw flattened 64x64 pixels (4096), no dimensionality reduction",
    "mask": {"rows": [MASK_ROWS.start, MASK_ROWS.stop - 1], "cols": [MASK_COLS.start, MASK_COLS.stop - 1],
             "value": 0.0},
    "single_tree": {**q1a["selected"], "selection": q1a["selection_rule"]},
    "bagging": {"n_trees": q1bc["selected_B"], "tree": q1bc["tree_params"],
                "bootstrap_size": 240, "selection": q1bc["selection_rule"]},
    "boosting": {"algorithm": "SAMME (own implementation)", **q2b["selected"],
                 "not_better_than_random": "eps >= 1-1/K -> reject, refit once with new seed, then stop",
                 "selection": q2b["selection_rule"]},
    "bias_variance": {"repetitions": q3["R"], "depth_study": q3["depths"],
                      "bagging_trees_in_depth_study": q3["B_depth"]},
}
(ROOT / "configs").mkdir(exist_ok=True)
with open(ROOT / "configs" / "frozen.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(json.dumps(cfg, indent=2))
