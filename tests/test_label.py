"""Gold-labelling app: blind items, autosave, token, and the 48-hour pass-2 rule."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from cpilens.label import store
from cpilens.label.app import make_handler


@pytest.fixture
def gold(tmp_path: Path) -> Path:
    shutil.copy(store.SAMPLE, tmp_path / "sample.csv")
    return tmp_path


@pytest.fixture
def server(gold: Path) -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(1, "tok", gold))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(base: str, body: dict[str, object], token: str = "tok") -> int:
    req = urllib.request.Request(
        f"{base}/api/label",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Token": token},
    )
    try:
        return int(urllib.request.urlopen(req).status)
    except urllib.error.HTTPError as e:
        return int(e.code)


def test_items_are_blind(server: str) -> None:
    init = json.load(urllib.request.urlopen(f"{server}/api/init"))
    assert len(init["items"]) == 600
    hidden = {"stratum", "silver_label", "incl_prob", "double", "source"}
    assert not hidden & set(init["items"][0])
    assert {"OOS", "UNK", "11522"} <= {x["label"] for x in init["labels"]}


def test_labels_autosave_and_latest_wins(server: str, gold: Path) -> None:
    init = json.load(urllib.request.urlopen(f"{server}/api/init"))
    o = init["items"][0]["order"]
    assert post(server, {"order": o, "label": "11122"}, token="wrong") == 403
    assert post(server, {"order": o, "label": "99999"}) == 400
    assert post(server, {"order": o, "label": "11122", "seconds": 7}) == 200
    assert post(server, {"order": o, "label": "11320", "unsure": True}) == 200
    lab = store.read_labels(1, gold)
    assert lab.rows()[0][2:4] == ("11320", "True")
    assert len((gold / "labels_pass1.csv").read_text().splitlines()) == 3  # header + audit trail


def test_pass2_waits_48_hours_and_reorders(gold: Path) -> None:
    doubles = store.queue(1, gold=gold).filter(pl.col("double"))
    now = datetime.now(UTC)
    for r in doubles.rows(named=True):
        store.append_label(
            1,
            {
                **r,
                "label": "11600",
                "unsure": False,
                "seconds": 5,
                "labelled_utc": now.isoformat(timespec="seconds"),
            },
            gold,
        )
    assert store.queue(2, now + timedelta(hours=47), gold).height == 0
    q2 = store.queue(2, now + timedelta(hours=49), gold)
    assert q2.height == doubles.height == 150
    assert q2["order"].to_list() != doubles["order"].to_list()
