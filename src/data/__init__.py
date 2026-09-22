"""Lightweight public imports for the data package.

Keep package import cheap so manifest/split planning does not require NumPy/Torch.
Heavy dataset symbols are resolved lazily on first access.
"""

from __future__ import annotations

from src.data.s3_downloader import S3DatasetDownloader
from src.data.hf_downloader import HFDatasetDownloader

__all__ = [
    "S3DatasetDownloader",
    "HFDatasetDownloader",
    "LipReadingDataset",
    "pad_collate_fn",
    "load_video_frames",
]


def __getattr__(name: str):
    if name in {"LipReadingDataset", "pad_collate_fn", "load_video_frames"}:
        from src.data.dataset import (
            LipReadingDataset,
            load_video_frames,
            pad_collate_fn,
        )

        mapping = {
            "LipReadingDataset": LipReadingDataset,
            "pad_collate_fn": pad_collate_fn,
            "load_video_frames": load_video_frames,
        }
        return mapping[name]
    raise AttributeError(name)
