"""Training constants of the classifiers, kept free of PyTorch so that the report (which
quotes them) runs without the ``ml`` dependency group."""

from __future__ import annotations

from dataclasses import dataclass

BUDGETS = (100, 250, 500, 1000, "all")
MLP_HIDDEN = 512
KNN_KS = (1, 3, 5, 10, 20, 50)

FT_HF = "FacebookAI/xlm-roberta-base"
FT_MAX_LEN = 64  # tokens; product texts are short
FT_EPOCHS = 4
FT_BATCH = 64
FT_LR = 3e-5


@dataclass
class TrainCfg:
    hidden: int = 0  # 0 = multinomial logistic regression
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 512
    min_steps: int = 50  # per evaluation round, so small training sets still get trained
    max_rounds: int = 40
    patience: int = 3
    seed: int = 20261027
