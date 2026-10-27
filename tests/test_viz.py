"""Figure helpers."""

from __future__ import annotations


def test_squarify_fills_rect():
    from cpilens.viz.treemap import squarify

    vals = [6, 6, 4, 3, 2, 2, 1]
    rs = squarify(vals, (0, 0, 6, 4))
    assert len(rs) == len(vals)
    for v, (_, _, w, h) in zip(vals, rs, strict=True):
        assert abs(w * h - v) < 1e-9
    assert all(
        x >= -1e-9 and y >= -1e-9 and x + w <= 6 + 1e-9 and y + h <= 4 + 1e-9 for x, y, w, h in rs
    )
