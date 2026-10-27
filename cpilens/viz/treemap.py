"""Squarified treemap layout (Bruls, Huizing and van Wijk, 2000), no dependencies."""

from __future__ import annotations

from collections.abc import Sequence

Rect = tuple[float, float, float, float]  # x, y, width, height


def _worst(row: list[float], side: float) -> float:
    s = sum(row)
    return max(max(side**2 * r / s**2, s**2 / (side**2 * r)) for r in row)


def squarify(values: Sequence[float], rect: Rect) -> list[Rect]:
    """Rectangles for ``values`` (sorted descending for best aspect ratios), filling ``rect``
    with areas proportional to the values, in input order."""
    x, y, w, h = rect
    total = sum(values)
    areas = [v * w * h / total for v in values]
    out: list[Rect] = []
    i = 0
    while i < len(areas):
        side = min(w, h)
        row = [areas[i]]
        i += 1
        while i < len(areas) and _worst([*row, areas[i]], side) <= _worst(row, side):
            row.append(areas[i])
            i += 1
        s = sum(row)
        if w >= h:  # lay the row out as a column on the left
            cw = s / h
            yy = y
            for a in row:
                out.append((x, yy, cw, a / cw))
                yy += a / cw
            x, w = x + cw, w - cw
        else:  # as a row along the bottom
            rh = s / w
            xx = x
            for a in row:
                out.append((xx, y, a / rh, rh))
                xx += a / rh
            y, h = y + rh, h - rh
    return out
