from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetMeta:
    dtype: str


def load_meta(meta_path: Path) -> DatasetMeta:
    obj = json.loads(meta_path.read_text(encoding="utf-8"))
    return DatasetMeta(dtype=str(obj["dtype"]))


class BlockDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Causal LM blocks:
      x = ids[i : i+T]
      y = ids[i+1 : i+T+1]
    """

    def __init__(
        self,
        bin_path: Path,
        *,
        dtype: np.dtype,
        block_size: int,
        stride: Optional[int] = None,
    ) -> None:
        self.bin_path = bin_path
        self.block_size = int(block_size)
        self.stride = int(stride) if stride is not None else int(block_size)

        self._data = np.memmap(bin_path, mode="r", dtype=dtype)
        self._n = int(self._data.shape[0])

        if self._n < self.block_size + 1:
            raise ValueError(
                f"{bin_path} has {self._n} tokens; need at least block_size+1={self.block_size + 1}"
            )

        # valid starting positions with step=stride
        self._num = 1 + (self._n - (self.block_size + 1)) // self.stride

    def __len__(self) -> int:
        return self._num

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = idx * self.stride

        # convert to int64 for embedding lookup
        x_np = np.asarray(self._data[i : i + self.block_size], dtype=np.int64)
        y_np = np.asarray(self._data[i + 1 : i + self.block_size + 1], dtype=np.int64)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        return x, y


def make_loaders(
    out_dir: Path,
    *,
    block_size: int,
    batch_size: int,
    stride: Optional[int] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    meta = load_meta(out_dir / "dataset_meta.json")
    dtype = np.dtype(meta.dtype)

    train_ds = BlockDataset(
        out_dir / "train.bin",
        dtype=dtype,
        block_size=block_size,
        stride=stride,
    )
    val_ds = BlockDataset(
        out_dir / "val.bin",
        dtype=dtype,
        block_size=block_size,
        stride=stride,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
