import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from common.config import (SEED, DATA_DIR, SPLIT_PATH, FIG_DIR, N_CLASSES,
                           N_TRAIN, N_VAL, N_TEST, MASK_ROWS, MASK_COLS)


def _fetch():
    faces = fetch_olivetti_faces(data_home=DATA_DIR)
    return faces.images, faces.target


def make_split():
    _, y = _fetch()
    rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)
        train_idx.extend(idx[:N_TRAIN])
        val_idx.extend(idx[N_TRAIN:N_TRAIN + N_VAL])
        test_idx.extend(idx[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST])
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SPLIT_PATH, seed=SEED,
             train=np.array(train_idx), val=np.array(val_idx), test=np.array(test_idx))
    return SPLIT_PATH


def mask_images(images):
    out = images.copy()
    out[:, MASK_ROWS, MASK_COLS] = 0.0
    return out


def load_data():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError("Saved split missing, run `python -m common.data` first")
    images, y = _fetch()
    split = np.load(SPLIT_PATH)
    assert int(split["seed"]) == SEED, "split was made with a different seed"

    d = {}
    for name in ("train", "val", "test"):
        idx = split[name]
        d[f"idx_{name}"] = idx
        d[f"X_{name}"] = images[idx].reshape(len(idx), -1)
        d[f"y_{name}"] = y[idx]
    d["X_test_masked"] = mask_images(images[split["test"]]).reshape(len(split["test"]), -1)
    d["images"] = images
    return d


def _sanity_figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    for j in range(4):
        axes[0, j].imshow(d["X_test"][j].reshape(64, 64), cmap="gray")
        axes[0, j].set_title(f"clean, id {d['y_test'][j]}", fontsize=8)
        axes[1, j].imshow(d["X_test_masked"][j].reshape(64, 64), cmap="gray")
        axes[1, j].set_title("masked", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "masked_examples.png", dpi=150)


if __name__ == "__main__":
    path = make_split()
    d = load_data()
    print(f"seed = {SEED}, split saved to {path}")
    for name in ("train", "val", "test"):
        X, yy = d[f"X_{name}"], d[f"y_{name}"]
        counts = np.bincount(yy, minlength=N_CLASSES)
        print(f"{name:5s}: X {X.shape}, per-identity counts min/max = {counts.min()}/{counts.max()}")
    all_idx = np.concatenate([d["idx_train"], d["idx_val"], d["idx_test"]])
    assert len(np.unique(all_idx)) == 400
    print("masked block mean (should be 0):", d["X_test_masked"].reshape(-1, 64, 64)[:, 24:40, 24:40].mean())
    _sanity_figure(d)
    print("sanity figure -> results/figures/masked_examples.png")
