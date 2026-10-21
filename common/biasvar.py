import numpy as np


def bias_variance(P, y):
    R, N, K = P.shape
    E = np.eye(K)[y]
    pbar = P.mean(axis=0)
    bias2 = float(np.mean(np.sum((pbar - E) ** 2, axis=1)))
    var = float(np.mean(np.sum((P - pbar) ** 2, axis=2)))
    mse = float(np.mean(np.sum((P - E) ** 2, axis=2)))
    assert np.isclose(mse, bias2 + var), (mse, bias2, var)

    preds = P.argmax(axis=2)
    acc = (preds == y).mean(axis=1)
    mode = np.array([np.bincount(preds[:, i], minlength=K).argmax() for i in range(N)])
    return {
        "bias2": bias2, "variance": var, "mse": mse,
        "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
        "acc_per_rep": acc,
        "label_flip_rate": float(np.mean(preds != mode)),
        "mean_max_prob": float(P.max(axis=2).mean()),
    }
