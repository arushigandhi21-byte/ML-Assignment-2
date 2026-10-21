import numpy as np
from sklearn.tree import DecisionTreeClassifier

EPS_FLOOR = 1e-10


def samme_update(w, y, pred, K):
    miss = (pred != y)
    eps = float(np.sum(w[miss]) / np.sum(w))
    if eps >= 1.0 - 1.0 / K:
        return {"eps": eps, "alpha": None, "w_new": w.copy(), "miss": miss, "status": "rejected"}
    status = "ok"
    if eps <= 0.0:
        eps, status = EPS_FLOOR, "perfect"
    alpha = np.log((1.0 - eps) / eps) + np.log(K - 1.0)
    w_new = w * np.exp(alpha * miss)
    w_new = w_new / w_new.sum()
    return {"eps": eps, "alpha": float(alpha), "w_new": w_new, "miss": miss, "status": status}


class SAMME:
    def __init__(self, n_rounds, max_depth, seed, n_classes):
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.seed = seed
        self.K = n_classes

    def _new_tree(self, rng):
        return DecisionTreeClassifier(max_depth=self.max_depth,
                                      random_state=int(rng.integers(2**31 - 1)))

    def fit(self, X, y, track=True):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        w = np.full(n, 1.0 / n)
        self.learners, self.alphas, self.eps = [], [], []
        self.events = []
        self.w_history = [w.copy()]
        self.train_pred_history = []
        for t in range(self.n_rounds):
            for attempt in range(2):
                tree = self._new_tree(rng)
                tree.fit(X, y, sample_weight=w)
                pred = tree.predict(X)
                upd = samme_update(w, y, pred, self.K)
                if upd["status"] != "rejected":
                    break
                self.events.append({"round": t, "attempt": attempt, "event": "rejected",
                                    "eps": upd["eps"]})
            if upd["status"] == "rejected":
                self.events.append({"round": t, "event": "stopped: two rejected learners in a row"})
                break
            self.learners.append(tree)
            self.alphas.append(upd["alpha"])
            self.eps.append(upd["eps"])
            w = upd["w_new"]
            if track:
                self.w_history.append(w.copy())
                self.train_pred_history.append(pred)
            if upd["status"] == "perfect":
                self.events.append({"round": t, "event": "perfect learner (eps=0), stopped"})
                break
        self.alphas = np.array(self.alphas)
        self.eps = np.array(self.eps)
        self.w_history = np.array(self.w_history)
        self.train_pred_history = np.array(self.train_pred_history)
        return self

    def learner_predictions(self, X, n_rounds=None):
        return np.stack([h.predict(X) for h in self.learners[:n_rounds]])

    def scores(self, X, n_rounds=None, P=None):
        if P is None:
            P = self.learner_predictions(X, n_rounds)
        P = P[:n_rounds]
        S = np.zeros((P.shape[1], self.K))
        for a, p in zip(self.alphas[:len(P)], P):
            S[np.arange(len(p)), p] += a
        return S

    def predict_proba(self, X, n_rounds=None, P=None):
        S = self.scores(X, n_rounds, P)
        return S / S.sum(axis=1, keepdims=True)

    def predict(self, X, n_rounds=None, P=None):
        return self.scores(X, n_rounds, P).argmax(axis=1)
