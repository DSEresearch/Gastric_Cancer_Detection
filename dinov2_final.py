#!/usr/bin/env python3
"""
dinov2_level0_ddp.py

DINOv2 feature extraction / SSL / supervised fine-tuning from clean Level-0 Zarr patches.

Input:
    clean_manifest.csv

Required manifest columns:
    slide_id,zarr_path,array_path,x0,y0,x1,y1,tile_size

Recommended 8-GPU feature extraction command:

python3.11 -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  dinov2_level0_ddp.py \
  --manifest ./clean_manifest.csv \
  --output-dir ./dinov2_features \
  --mode extract \
  --backbone dinov2_vitb14 \
  --dinov2-repo /home/jovyan/dinov2 \
  --tile-size 512 \
  --batch-size 32 \
  --num-workers 0

After successful test, increase:
  --num-workers 4
then:
  --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import zarr

from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)

        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int):
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def open_zarr_group(zarr_path: str):
    if hasattr(zarr, "open"):
        return zarr.open(zarr_path, mode="r")

    if hasattr(zarr, "open_group"):
        return zarr.open_group(zarr_path, mode="r")

    raise RuntimeError(
        "The imported zarr module has neither zarr.open nor zarr.open_group. "
        f"Imported zarr path: {getattr(zarr, '__file__', 'unknown')}"
    )


def read_zarr_region_rgb(arr, x: int, y: int, width: int, height: int) -> np.ndarray:
    shape = arr.shape

    if len(shape) == 5:
        patch = arr[0, :3, 0, y:y + height, x:x + width]
        patch = np.asarray(patch)
        patch = np.transpose(patch, (1, 2, 0))

    elif len(shape) == 3 and shape[-1] in (3, 4):
        patch = arr[y:y + height, x:x + width, :3]
        patch = np.asarray(patch)

    elif len(shape) == 3 and shape[0] in (3, 4):
        patch = arr[:3, y:y + height, x:x + width]
        patch = np.asarray(patch)
        patch = np.transpose(patch, (1, 2, 0))

    else:
        raise ValueError(f"Unsupported Zarr array shape: {shape}")

    if patch.ndim != 3 or patch.shape[2] < 3:
        raise ValueError(f"Invalid RGB patch shape: {patch.shape}")

    if patch.shape[0] != height or patch.shape[1] != width:
        raise ValueError(
            f"Patch shape mismatch. Expected ({height}, {width}, 3), got {patch.shape}"
        )

    return patch[:, :, :3].astype(np.uint8)


class ZarrCleanPatchDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        tile_size: int = 512,
        mode: str = "extract",
        label_column: Optional[str] = None,
        transform=None,
        transform_ssl=None,
    ):
        self.df = pd.read_csv(
            manifest_csv,
            dtype={"slide_id": str},
            low_memory=False,
        )

        required = {"slide_id", "zarr_path", "array_path", "x0", "y0"}
        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        self.tile_size = tile_size
        self.mode = mode
        self.label_column = label_column
        self.transform = transform
        self.transform_ssl = transform_ssl

        # Critical fix:
        # Each DataLoader worker gets its own Dataset copy.
        # Therefore, the Zarr cache must exist on every worker copy.
        self._cache: Dict[Tuple[str, str], Any] = {}

        if self.mode == "supervised":
            if not label_column or label_column not in self.df.columns:
                raise ValueError(
                    "--mode supervised requires --label-column and that column must exist."
                )

            labels = sorted(self.df[label_column].astype(str).unique().tolist())
            self.label_to_id = {label: i for i, label in enumerate(labels)}
        else:
            self.label_to_id = {}

    def __len__(self):
        return len(self.df)

    def _get_array(self, zarr_path: str, array_path: str):
        key = (zarr_path, array_path)

        # Defensive fix for DataLoader worker serialization/rehydration.
        if not hasattr(self, "_cache"):
            self._cache = {}

        if key not in self._cache:
            root = open_zarr_group(zarr_path)
            self._cache[key] = root[array_path]

        return self._cache[key]

    def _read_image(self, row) -> Image.Image:
        zarr_path = str(row["zarr_path"])
        array_path = str(row["array_path"])
        x = int(row["x0"])
        y = int(row["y0"])

        if "tile_size" in row and not pd.isna(row["tile_size"]):
            tile_size = int(row["tile_size"])
        else:
            tile_size = self.tile_size

        if tile_size <= 0:
            tile_size = self.tile_size

        arr = self._get_array(zarr_path, array_path)

        patch = read_zarr_region_rgb(
            arr=arr,
            x=x,
            y=y,
            width=tile_size,
            height=tile_size,
        )

        #return Image.fromarray(patch, mode="RGB")
        return Image.fromarray(patch).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self._read_image(row)

        meta = {
            "slide_id": str(row["slide_id"]),
            "x": int(row["x0"]),
            "y": int(row["y0"]),
            "index": int(idx),
        }

        if self.mode == "ssl":
            v1 = self.transform_ssl(image)
            v2 = self.transform_ssl(image)
            return v1, v2, meta

        if self.mode == "extract":
            x = self.transform(image)
            return x, meta

        if self.mode == "supervised":
            x = self.transform(image)
            label_str = str(row[self.label_column])
            y = self.label_to_id[label_str]
            return x, torch.tensor(y, dtype=torch.long), meta

        raise ValueError(f"Unknown mode: {self.mode}")


def get_eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def get_ssl_transform(tile_size: int):
    return T.Compose([
        T.RandomResizedCrop(
            size=tile_size,
            scale=(0.60, 1.00),
            ratio=(0.90, 1.10),
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.15, 0.15, 0.10, 0.05)], p=0.5),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_dinov2(backbone_name: str, dinov2_repo: Optional[str] = None):
    if dinov2_repo:
        return torch.hub.load(
            dinov2_repo,
            backbone_name,
            source="local",
        )

    return torch.hub.load(
        "facebookresearch/dinov2",
        backbone_name,
    )


class DINOv2WithProjection(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        dinov2_repo: Optional[str] = None,
        proj_dim: int = 256,
        train_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = load_dinov2(backbone_name, dinov2_repo)
        embed_dim = self.backbone.embed_dim

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out

    def forward(self, x):
        feats = self.forward_features(x)
        z = self.proj(feats)
        return F.normalize(z, dim=-1)


class DINOv2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "dinov2_vits14",
        dinov2_repo: Optional[str] = None,
        train_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = load_dinov2(backbone_name, dinov2_repo)
        embed_dim = self.backbone.embed_dim
        self.head = nn.Linear(embed_dim, num_classes)

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward_features(self, x):
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            return out["x_norm_clstoken"]
        return out

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)


def nt_xent_loss(z1, z2, temperature: float = 0.2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    if is_dist():
        z1_all = [torch.zeros_like(z1) for _ in range(get_world_size())]
        z2_all = [torch.zeros_like(z2) for _ in range(get_world_size())]

        dist.all_gather(z1_all, z1.detach())
        dist.all_gather(z2_all, z2.detach())

        z1_all[get_rank()] = z1
        z2_all[get_rank()] = z2

        z1_all = torch.cat(z1_all, dim=0)
        z2_all = torch.cat(z2_all, dim=0)
    else:
        z1_all = z1
        z2_all = z2

    logits_12 = z1 @ z2_all.T / temperature
    logits_21 = z2 @ z1_all.T / temperature

    batch_size = z1.shape[0]
    labels = torch.arange(batch_size, device=z1.device) + get_rank() * batch_size

    return 0.5 * (
        F.cross_entropy(logits_12, labels) +
        F.cross_entropy(logits_21, labels)
    )


def save_checkpoint(path: Path, model, optimizer, scaler, epoch, args):
    model_to_save = model.module if isinstance(model, DDP) else model

    ckpt = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        "args": vars(args),
    }

    torch.save(ckpt, path)


@torch.no_grad()
def extract_features(args, device):
    dataset = ZarrCleanPatchDataset(
        manifest_csv=args.manifest,
        tile_size=args.tile_size,
        mode="extract",
        transform=get_eval_transform(),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,
        drop_last=False,
    ) if is_dist() else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=False,
    )

    model = load_dinov2(args.backbone, args.dinov2_repo).to(device)
    model.eval()

    features = []
    metas = []

    for step, batch in enumerate(loader):
        x, meta = batch
        x = x.to(device, non_blocking=True)

        with autocast("cuda",
            enabled=args.amp,
            dtype=torch.bfloat16 if args.bf16 else torch.float16,
        ):
            out = model.forward_features(x)
            if isinstance(out, dict):
                feats = out["x_norm_clstoken"]
            else:
                feats = out

        features.append(feats.detach().cpu())

        batch_size = x.size(0)

        for i in range(batch_size):
            metas.append({
                "slide_id": meta["slide_id"][i],
                "x": int(meta["x"][i]),
                "y": int(meta["y"][i]),
                "index": int(meta["index"][i]),
            })

        if is_main() and step % args.log_every == 0:
            print(f"extract step={step}/{len(loader)}", flush=True)

    if features:
        features = torch.cat(features, dim=0)
    else:
        features = torch.empty(0)

    out_path = args.output_dir / f"features_rank{get_rank():02d}.pt"

    torch.save(
        {
            "features": features,
            "meta": metas,
            "rank": get_rank(),
            "world_size": get_world_size(),
            "backbone": args.backbone,
        },
        out_path,
    )

    if is_main():
        print(f"Saved rank-wise feature files to {args.output_dir}", flush=True)


def train_ssl(args, device):
    dataset = ZarrCleanPatchDataset(
        manifest_csv=args.manifest,
        tile_size=args.tile_size,
        mode="ssl",
        transform_ssl=get_ssl_transform(args.tile_size),
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=True,
    ) if is_dist() else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )

    model = DINOv2WithProjection(
        backbone_name=args.backbone,
        dinov2_repo=args.dinov2_repo,
        proj_dim=args.proj_dim,
        train_backbone=not args.freeze_backbone,
    ).to(device)

    if is_dist():
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        loss_sum = 0.0
        steps = 0
        start = time.time()

        for step, batch in enumerate(loader):
            x1, x2, _meta = batch
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda",
                enabled=args.amp,
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
            ):
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach())
            steps += 1

            if is_main() and step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={float(loss.detach()):.4f}",
                    flush=True,
                )

        avg_loss = loss_sum / max(1, steps)

        if is_main():
            elapsed = time.time() - start
            print(
                f"[epoch {epoch}] avg_loss={avg_loss:.4f} time={elapsed:.1f}s",
                flush=True,
            )

            save_checkpoint(
                args.output_dir / f"ssl_checkpoint_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
            )


def train_supervised(args, device):
    dataset = ZarrCleanPatchDataset(
        manifest_csv=args.manifest,
        tile_size=args.tile_size,
        mode="supervised",
        label_column=args.label_column,
        transform=get_eval_transform(),
    )

    num_classes = len(dataset.label_to_id)

    if is_main():
        with open(args.output_dir / "label_map.json", "w") as f:
            json.dump(dataset.label_to_id, f, indent=2)

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=False,
    ) if is_dist() else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=False,
    )

    model = DINOv2Classifier(
        num_classes=num_classes,
        backbone_name=args.backbone,
        dinov2_repo=args.dinov2_repo,
        train_backbone=not args.freeze_backbone,
    ).to(device)

    if is_dist():
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0

        for step, batch in enumerate(loader):
            x, y, _meta = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda",
                enabled=args.amp,
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
            ):
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            loss_sum += float(loss.detach()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().detach())
            seen += x.size(0)

            if is_main() and step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={float(loss.detach()):.4f}",
                    flush=True,
                )

        if is_dist():
            tensor = torch.tensor(
                [loss_sum, correct, seen],
                dtype=torch.float64,
                device=device,
            )

            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            loss_sum, correct, seen = tensor.tolist()

        if is_main():
            print(
                f"[epoch {epoch}] loss={loss_sum / max(1, seen):.4f} "
                f"acc={correct / max(1, seen):.4f}",
                flush=True,
            )

            save_checkpoint(
                args.output_dir / f"supervised_checkpoint_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                args,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 DDP pipeline for clean Level-0 Zarr patches."
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./dinov2_out"))

    parser.add_argument(
        "--mode",
        choices=["extract", "ssl", "supervised"],
        default="extract",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
    )

    parser.add_argument(
        "--dinov2-repo",
        type=str,
        default="/home/jovyan/dinov2",
        help="Local DINOv2 repo path. Use local repo to avoid torch.hub cache race.",
    )

    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Per-GPU DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--label-column", type=str, default=None)

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if is_main():
        print("DINOv2 Level-0 Zarr DDP Pipeline", flush=True)
        print(f"world_size={world_size}", flush=True)
        print(f"device={device}", flush=True)
        print(f"manifest={args.manifest}", flush=True)
        print(f"output_dir={args.output_dir}", flush=True)
        print(f"mode={args.mode}", flush=True)
        print(f"backbone={args.backbone}", flush=True)
        print(f"dinov2_repo={args.dinov2_repo}", flush=True)
        print(f"per_gpu_batch_size={args.batch_size}", flush=True)
        print(f"global_batch_size={args.batch_size * world_size}", flush=True)

        with open(args.output_dir / "run_args.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    try:
        if args.mode == "extract":
            extract_features(args, device)
        elif args.mode == "ssl":
            train_ssl(args, device)
        elif args.mode == "supervised":
            train_supervised(args, device)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
