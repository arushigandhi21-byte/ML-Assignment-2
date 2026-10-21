import numpy as np
from sklearn.tree import DecisionTreeClassifier

from joblib import Parallel, delayed

from common.config import N_CLASSES


def _fit_tree(X, y, params, random_state):
    return DecisionTreeClassifier(**params, random_state=random_state).fit(X, y)


class Bagging:
    def __init__(self, n_trees, tree_params, seed):
        self.n_trees = n_trees
        self.tree_params = dict(tree_params)
        self.seed = seed

    def fit(self, X, y, n_jobs=-1):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.boot_idx, states = [], []
        for _ in range(self.n_trees):
            self.boot_idx.append(rng.integers(0, n, size=n))
            states.append(int(rng.integers(2**31 - 1)))
        self.trees = Parallel(n_jobs=n_jobs)(
            delayed(_fit_tree)(X[idx], y[idx], self.tree_params, s)
            for idx, s in zip(self.boot_idx, states))
        self.n_train = n
        return self

    def tree_predictions(self, X, n_trees=None):
        trees = self.trees[:n_trees]
        return np.stack([t.predict(X) for t in trees])

    @staticmethod
    def votes_from_predictions(P):
        B, n = P.shape
        votes = np.zeros((n, N_CLASSES), dtype=int)
        for b in range(B):
            votes[np.arange(n), P[b]] += 1
        return votes

    def votes(self, X, n_trees=None):
        return self.votes_from_predictions(self.tree_predictions(X, n_trees))

    def predict_proba(self, X, n_trees=None):
        v = self.votes(X, n_trees)
        return v / v.sum(axis=1, keepdims=True)

    def predict(self, X, n_trees=None):
        return self.votes(X, n_trees).argmax(axis=1)

    def oob_mask(self, n_trees=None):
        m = np.ones((len(self.boot_idx[:n_trees]), self.n_train), dtype=bool)
        for b, idx in enumerate(self.boot_idx[:n_trees]):
            m[b, idx] = False
        return m


def vote_margin(votes):
    frac = votes / votes.sum(axis=1, keepdims=True)
    top2 = np.sort(frac, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def pairwise_disagreement(P):
    B = P.shape[0]
    if B < 2:
        return float("nan")
    votes = Bagging.votes_from_predictions(P)
    agree_pairs = (votes * (votes - 1) / 2).sum(axis=1)
    total_pairs = B * (B - 1) / 2
    return float(np.mean(1 - agree_pairs / total_pairs))


def oob_accuracy(model, X_train, y_train, n_trees=None):
    P = model.tree_predictions(X_train, n_trees)
    oob = model.oob_mask(n_trees)
    n = len(y_train)
    votes = np.zeros((n, N_CLASSES), dtype=int)
    for b in range(P.shape[0]):
        rows = np.flatnonzero(oob[b])
        votes[rows, P[b, rows]] += 1
    has_oob = votes.sum(axis=1) > 0
    acc = float(np.mean(votes[has_oob].argmax(axis=1) == y_train[has_oob]))
    return acc, float(has_oob.mean())
