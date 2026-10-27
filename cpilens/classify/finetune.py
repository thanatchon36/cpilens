"""Fine-tuned multilingual encoder (XLM-R base) with a classification head.

Same training pools, class weighting (1/sqrt frequency), validation-based early stopping and
temperature scaling as the embedding classifiers, so the comparison differs only in the
model. Mixed precision, max 64 tokens (product texts are short), AdamW with linear warm-up
and decay. The selected checkpoint (best validation loss) is kept under ``.cache/ft``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import torch
from torch import nn

from cpilens.classify import embedclf as ec
from cpilens.classify import experiments as ex
from cpilens.classify.config import FT_BATCH, FT_EPOCHS, FT_HF, FT_LR, FT_MAX_LEN
from cpilens.classify.data import corpus
from cpilens.paths import CACHE

HF = FT_HF
FT_DIR = CACHE / "ft"
MAX_LEN = FT_MAX_LEN


def _batches(tok: object, texts: list[str], bs: int, max_len: int = MAX_LEN):  # type: ignore[no-untyped-def]
    for s in range(0, len(texts), bs):
        yield tok(
            texts[s : s + bs],
            padding=True,
            truncation=True,
            max_length=max_len,  # type: ignore[operator]
            return_tensors="pt",
        ).to(ec.DEVICE)


def logits_for(model: nn.Module, tok: object, texts: list[str], bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in _batches(tok, texts, bs):
            out.append(model(**b).logits.float().cpu().numpy())
    return np.vstack(out)


def run(
    scenario: str = "nonthai+thaiweak+mocall",
    epochs: int = FT_EPOCHS,
    bs: int = FT_BATCH,
    lr: float = FT_LR,
    seed: int = 20261027,
) -> dict[str, object]:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    c = corpus()
    cls = ex.classes()
    pool = dict(ex.scenarios(c))[scenario]
    tr, va = pool.filter(pl.col("split") == "train"), pool.filter(pl.col("split") == "val")
    ytr = ec.encode_labels(tr["label"].to_list(), cls)
    yva = ec.encode_labels(va["label"].to_list(), cls)
    tok = AutoTokenizer.from_pretrained(HF)
    model = AutoModelForSequenceClassification.from_pretrained(HF, num_labels=len(cls))
    model = model.to(ec.DEVICE)
    freq = np.bincount(ytr, minlength=len(cls)).astype(float)
    cw = np.where(freq > 0, 1 / np.sqrt(np.maximum(freq, 1)), 0.0)
    cw = cw / (cw * freq).sum() * freq.sum()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=ec.DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    texts, g = tr["text"].to_list(), np.random.default_rng(seed)
    out_dir = FT_DIR / f"xlmr__{scenario}"
    best, t0, history = float("inf"), time.time(), []
    for ep in range(epochs):
        model.train()
        order = g.permutation(len(texts))
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            b = tok(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(ec.DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(**b).logits.float(), torch.from_numpy(ytr[idx]).to(ec.DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        lva = logits_for(model, tok, va["text"].to_list())
        v = float(loss_fn(torch.from_numpy(lva).to(ec.DEVICE), torch.from_numpy(yva).to(ec.DEVICE)))
        history.append(v)
        print(f"epoch {ep + 1}: val loss {v:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if v < best - 1e-4:
            best = v
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
        else:
            break
    fit_s = time.time() - t0
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(ec.DEVICE)
    lva = logits_for(model, tok, va["text"].to_list())
    temp = ec.fit_temperature(lva, yva)
    ev = ex.eval_sets(c)
    t1 = time.time()
    prob = ec.softmax(logits_for(model, tok, ev["text"].to_list()), temp)
    pred_s = (time.time() - t1) / ev.height
    run_id = f"xlmr__finetune__{scenario}"
    out = ev.select("id", "eval_set", "label").with_columns(
        pl.Series("pred", [cls[i] for i in prob.argmax(1)]), pl.Series("conf", prob.max(1))
    )
    ex.PREDS.mkdir(parents=True, exist_ok=True)
    out.write_parquet(ex.PREDS / f"{run_id}.parquet")
    row = {
        "run": run_id,
        "emb": "xlmr",
        "clf": "finetune",
        "scenario": scenario,
        "n_train": tr.height,
        "n_val": va.height,
        "n_official": int((pool["label_kind"] == "official").sum()),
        "temperature": temp,
        "ece_val_raw": ec.ece(ec.softmax(lva), yva),
        "ece_val": ec.ece(ec.softmax(lva, temp), yva),
        "fit_s": fit_s,
        "pred_s_per_item": pred_s,
        "epochs_run": len(history),
        **ex.dev_metrics(out),
    }
    print(json.dumps(row, default=str), flush=True)
    return row
