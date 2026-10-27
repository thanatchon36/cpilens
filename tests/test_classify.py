import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cpilens.classify import embedclf as ec  # noqa: E402


def blobs(n_per: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.eye(3, 16) * 3
    x = np.vstack([c + rng.normal(0, 0.5, (n_per, 16)) for c in centres]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x, np.repeat(np.arange(3), n_per)


@pytest.mark.parametrize("hidden", [0, 32])
def test_small_training_sets_are_trained(hidden):
    # regression: 30 rows with batch 512 used to mean one step per epoch -> constant output
    xtr, ytr = blobs(10, 0)
    xva, yva = blobs(5, 1)
    net = ec.train_net(xtr, ytr, xva, yva, 3, ec.TrainCfg(hidden=hidden))
    pred = ec.net_logits(net, xva).argmax(1)
    assert (pred == yva).mean() > 0.9


def test_knn_and_temperature():
    xtr, ytr = blobs(10, 0)
    xva, yva = blobs(5, 1)
    k = ec.knn_tune(xtr, ytr, xva, yva, 3)
    lg = ec.knn_logits(xtr, ytr, xva, k, 3)
    assert (lg.argmax(1) == yva).mean() > 0.9
    t = ec.fit_temperature(lg, yva)
    assert t > 0
    p = ec.softmax(lg, t)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert 0 <= ec.ece(p, yva) <= 1
