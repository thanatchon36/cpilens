"""Local gold-labelling app: a stdlib HTTP server on 127.0.0.1 and one static page.

No model or silver suggestion is ever sent to the page: items carry only what a shopper
would see (names, brand, quantity, front image). The label list is the CPI scope table with
TPSO's own item headings as search aliases (taxonomy, not predictions).
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

import polars as pl

from cpilens.label import store
from cpilens.paths import INTERIM, TAXONOMY

ITEM_FIELDS = ["order", "barcode", "name_th", "name_en", "brands", "quantity", "image_url"]


def label_list() -> list[dict[str, Any]]:
    scope = pl.read_csv(TAXONOMY / "thcpi_scope.csv", schema_overrides={"label": pl.Utf8})
    aliases: dict[str, list[str]] = {}
    specs = INTERIM / "moc_specs.parquet"
    if specs.exists():
        top = (
            pl.read_parquet(specs)
            .filter(pl.col("item_th") != "")
            .group_by("code5", "item_th")
            .len()
            .sort("len", descending=True)
        )
        for code, item, _ in top.rows():
            aliases.setdefault(code, []).append(item)
    out = [
        {
            "label": r["label"],
            "name_th": r["name_th"] or "",
            "name_en": r["name_en"],
            "group": r["scope_group"],
            "includes": r["includes"] or "",
            "aliases": aliases.get(r["label"], [])[:20],
        }
        for r in scope.iter_rows(named=True)
    ]
    out.append(
        {
            "label": "UNK",
            "name_th": "ระบุไม่ได้",
            "name_en": "Cannot tell",
            "group": "special",
            "includes": store.SPECIAL["UNK"],
            "aliases": [],
        }
    )
    return out


def make_handler(pass_no: int, token: str, gold: Path = store.GOLD) -> type[BaseHTTPRequestHandler]:
    labels = label_list()
    valid = {x["label"] for x in labels}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/":
                page = files("cpilens.label").joinpath("index.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/init":
                q = store.queue(pass_no, gold=gold)
                done = store.read_labels(pass_no, gold).select("order", "label", "unsure")
                self._json(
                    {
                        "pass": pass_no,
                        "labels": labels,
                        "items": q.select(ITEM_FIELDS).to_dicts(),
                        "done": {
                            o: {"label": lab, "unsure": u == "True"} for o, lab, u in done.rows()
                        },
                    }
                )
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/api/label" or self.headers.get("X-Token") != token:
                self._send(403, b"forbidden", "text/plain")
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            items = {
                i["order"]: i
                for i in store.queue(pass_no, gold=gold).select(ITEM_FIELDS).to_dicts()
            }
            if body.get("order") not in items or body.get("label") not in valid:
                self._json({"error": "unknown item or label"}, 400)
                return
            store.append_label(
                pass_no,
                {
                    "order": body["order"],
                    "barcode": items[body["order"]]["barcode"],
                    "label": body["label"],
                    "unsure": bool(body.get("unsure")),
                    "seconds": round(float(body.get("seconds", 0)), 1),
                    "labelled_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                gold,
            )
            self._json({"ok": True})

        def log_message(self, format: str, *args: object) -> None:  # quiet console
            pass

    return Handler


def serve(pass_no: int = 1, port: int = 8765) -> None:
    token = secrets.token_urlsafe(16)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(pass_no, token))
    n = store.queue(pass_no).height
    print(f"Gold labelling, pass {pass_no}: {n} products available")
    print(f"Open  http://127.0.0.1:{port}/?token={token}   (Ctrl+C to stop; every label autosaves)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
