"""ViT-Tiny / CIFAR-10 demo trainer for Short Game's live orchestration demo.

The baseline config is deliberately broken (fp32, batch_size=8, num_workers=0,
no AMP) so the orchestrator can detect waste and propose fixes mid-run.

Per-step JSON is emitted to stdout so the orchestrator can parse it over SSH.
SIGUSR1 triggers a checkpoint + graceful restart from the current weights.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CKPT_DIR = Path("/workspace/ckpt")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_PATH = CKPT_DIR / "latest.pt"

_SIGUSR_RECEIVED = {"v": False}


def _sigusr1_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    _SIGUSR_RECEIVED["v"] = True


signal.signal(signal.SIGUSR1, _sigusr1_handler)


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build ViT-Tiny via timm if available, else a small ViT fallback."""
    try:
        import timm

        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=10,
            img_size=cfg.get("img_size", 224),
        )
        if cfg.get("attn_impl", "sdpa") == "sdpa":
            # timm honors torch's SDPA by default on recent versions.
            pass
        return model
    except Exception:  # pragma: no cover
        # Tiny fallback so the script still runs if timm is missing.
        class TinyCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(64, 10),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c(x)

        return TinyCNN()


def make_loader(cfg: dict[str, Any]) -> DataLoader:
    tfm = transforms.Compose(
        [
            transforms.Resize(cfg.get("img_size", 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    root = cfg.get("data_root", "/workspace/data")
    Path(root).mkdir(parents=True, exist_ok=True)
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    return DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", False)),
        persistent_workers=bool(cfg.get("persistent_workers", False)) and int(cfg.get("num_workers", 0)) > 0,
        drop_last=True,
    )


def _dtype_from_cfg(cfg: dict[str, Any]) -> torch.dtype:
    d = str(cfg.get("dtype", "fp32")).lower()
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(d, torch.float32)


def _fake_util(cfg: dict[str, Any], step_ms: float) -> dict[str, float]:
    """Return heuristic utilization numbers so the orchestrator can score forks.

    We don't have DCGM available in the demo image — emit deterministic values
    that respond to config choices so detector rules fire realistically.
    """
    dtype = str(cfg.get("dtype", "fp32")).lower()
    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 0))
    amp = bool(cfg.get("amp", False))

    # pipe_tensor_active: high only with fp16/bf16 or AMP
    if dtype in ("fp16", "bf16") or amp:
        pipe = 52.0
    else:
        pipe = 12.0
    # sm_active: limited by tiny batch and/or slow dataloader
    sm = 80.0
    if bs <= 16:
        sm -= 25.0
    if nw == 0:
        sm -= 20.0
    sm = max(8.0, min(95.0, sm + (5.0 if amp else 0.0)))
    # dram_active: roughly scales with batch/dtype width
    dram = 30.0 + (10.0 if bs >= 32 else 0.0) + (5.0 if dtype == "fp32" else 0.0)
    return {"sm_active": sm, "pipe_tensor_active": pipe, "dram_active": dram}


def _log_step(step: int, step_ms: float, loss: float, cfg: dict[str, Any]) -> None:
    util = _fake_util(cfg, step_ms)
    rec = {
        "step": step,
        "step_ms": round(step_ms, 3),
        "loss": round(loss, 4),
        **{k: round(v, 2) for k, v in util.items()},
    }
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    torch.save(
        {"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step},
        CKPT_PATH,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    try:
        optimizer.load_state_dict(state["opt"])
    except Exception:
        pass
    return int(state.get("step", 0))


def train(cfg: dict[str, Any], resume: str | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_cfg(cfg)
    amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    model = build_model(cfg).to(device=device, dtype=dtype if not amp else torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    loader = make_loader(cfg)
    start_step = 0
    if resume and os.path.exists(resume):
        start_step = load_checkpoint(resume, model, optimizer)
        sys.stdout.write(json.dumps({"event": "resumed", "from_step": start_step}) + "\n")
        sys.stdout.flush()

    scaler = torch.cuda.amp.GradScaler(enabled=amp and dtype == torch.float16)
    total_steps = int(cfg.get("total_steps", 600))
    step = start_step
    data_iter = iter(loader)
    while step < total_steps:
        if _SIGUSR_RECEIVED["v"]:
            save_checkpoint(model, optimizer, step)
            sys.stdout.write(json.dumps({"event": "checkpoint", "step": step}) + "\n")
            sys.stdout.flush()
            # Exit 0 so the orchestrator can restart from latest.pt with new config.
            return
        t0 = time.perf_counter()
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16):
                out = model(x)
                loss = F.cross_entropy(out, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            x_in = x.to(dtype) if dtype != torch.float32 else x
            out = model(x_in)
            loss = F.cross_entropy(out.float(), y)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_ms = (time.perf_counter() - t0) * 1000.0
        if not math.isfinite(loss.item()):
            loss_val = 99.0
        else:
            loss_val = float(loss.item())
        _log_step(step, step_ms, loss_val, cfg)
        step += 1
    save_checkpoint(model, optimizer, step)
    sys.stdout.write(json.dumps({"event": "done", "step": step}) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.resume)


if __name__ == "__main__":
    main()
