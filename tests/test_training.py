import numpy as np

from scripts.train_model import train_probe


def test_probe_selects_c_on_validation_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(200, 8)); ytr = (Xtr[:, 0] > 0).astype(int)
    Xva = rng.normal(size=(80, 8)); yva = (Xva[:, 0] > 0).astype(int)
    model, c, scores = train_probe(Xtr, ytr, Xva, yva)
    assert c in scores and scores[c] == max(scores.values()) and scores[c] > 0.9


def test_probe_skips_single_class_split():
    X = np.zeros((30, 4))
    assert train_probe(X, np.zeros(30, int), X, np.ones(30, int)) is None
