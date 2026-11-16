"""Training plumbing shared by the letter CNN and the recognizer: config, device, seeds,
resumable checkpoints and logging."""
from __future__ import annotations

import copy
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from common.paths import load_yaml, REPO_ROOT


def load_step2_config(small: bool = False) -> dict:
    cfg = load_yaml(REPO_ROOT / "configs" / "step2_cnn.yaml")
    if small:  # deep-merge the small: block for quick tests
        for section, over in cfg["small"].items():
            cfg[section] = {**cfg[section], **over}
    cfg["small_mode"] = small
    return cfg


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"{name}.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


class Checkpointer:
    """Saves last.pt every epoch (to resume) and best.pt when the monitored metric improves."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, model, optimizer, scheduler, epoch: int, best: float, extra: dict | None = None):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best": best,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
            "extra": extra or {},
        }
        tmp = self.dir / f"{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(self.dir / f"{name}.pt")  # atomic: an interrupted save never corrupts the file

    def load(self, name: str, model, optimizer=None, scheduler=None, restore_rng: bool = False) -> dict | None:
        """Load onto CPU first (load_state_dict copies into the model's device; RNG state must stay on CPU)."""
        path = self.dir / f"{name}.pt"
        if not path.exists():
            return None
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        rng = state.get("rng")
        if restore_rng and rng:  # only when resuming training
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
        return state


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
