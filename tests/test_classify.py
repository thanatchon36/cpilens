import numpy as np
import polars as pl
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


def test_moc_split_holds_out_subitems_and_covers_every_item():
    from cpilens.classify import data
    from cpilens.paths import INTERIM

    if not (INTERIM / "moc_specs.parquet").exists():
        pytest.skip("needs the official specifications")
    raw = pl.read_parquet(INTERIM / "moc_specs.parquet").select("spec_id", "item_code")
    d = data.moc_rows().with_columns(pl.col("id").str.strip_prefix("moc:").alias("spec_id"))
    d = d.join(raw, on="spec_id")
    test = d.filter(pl.col("split") == "test")
    assert set(test["label"]) == set(d["label"])  # every item (and OOS) has test rows
    multi = d.group_by("label").agg(pl.col("item_code").n_unique().alias("n"))
    multi = multi.filter(pl.col("n") >= 2)["label"]
    per_code = (
        d.filter(pl.col("label").is_in(multi))
        .group_by("item_code")
        .agg(pl.col("split").n_unique().alias("k"))
    )
    assert (per_code["k"] == 1).all()  # a sub-item sits in one split only
    key = pl.col("text").str.to_lowercase().str.replace_all(r"\s+", "")
    train_keys = d.filter(pl.col("split") != "test").with_columns(key.alias("k")).select("k")
    assert test.with_columns(key.alias("k")).join(train_keys.unique(), on="k").height == 0
