"""
Dataset module for Zero-Shot Text Classification.

Implements data loading, preprocessing, negative sampling, and a
PyTorch Dataset/DataLoader wrapper for the BiEncoder training pipeline.
"""

import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Negative Sampling
# ---------------------------------------------------------------------------

def negative_sampling(
    batch_labels: List[List[str]],
    all_labels: List[str],
    max_num_negatives: int = 5,
) -> List[List[str]]:
    """
    Sample hard negatives for each example in the batch.

    For each sample, randomly selects between 1 and `max_num_negatives`
    labels from the global label pool that are NOT positive labels for
    that sample.

    Args:
        batch_labels: List of positive label lists for each sample.
        all_labels: The complete pool of all unique labels in the dataset.
        max_num_negatives: Maximum number of negatives per sample.

    Returns:
        List of negative label lists, one per sample.
    """
    num_negatives = random.randint(1, max_num_negatives)
    negative_samples = []
    for labels in batch_labels:
        label_set = set(labels)
        candidates = [l for l in all_labels if l not in label_set]
        # Guard against edge case where all labels are positive
        k = min(num_negatives, len(candidates))
        neg = random.sample(candidates, k) if k > 0 else []
        negative_samples.append(neg)
    return negative_samples


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class ZeroShotDataset(Dataset):
    """
    PyTorch Dataset for zero-shot text classification.

    Each sample consists of:
        - text: the input sentence.
        - candidate_labels: positive labels + sampled negatives.
        - targets: binary tensor marking positive labels.

    Args:
        data_path (str): Path to the JSON data file.
        max_num_labels (int): Maximum candidate labels per sample (pad/truncate).
        max_num_negatives (int): Max negative labels to add per sample.
        split (str): 'train' or 'val'.
        train_split (float): Fraction of data used for training.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        data_path: str,
        max_num_labels: int = 5,
        max_num_negatives: int = 5,
        split: str = "train",
        train_split: float = 0.9,
        seed: int = 42,
    ):
        self.max_num_labels = max_num_labels
        self.max_num_negatives = max_num_negatives

        with open(data_path, "r") as f:
            raw_data = json.load(f)

        # Build the global label vocabulary
        self.all_labels: List[str] = sorted(
            set(label for item in raw_data for label in item["labels"])
        )

        # Train / val split
        random.seed(seed)
        indices = list(range(len(raw_data)))
        random.shuffle(indices)
        cutoff = int(len(indices) * train_split)
        split_indices = indices[:cutoff] if split == "train" else indices[cutoff:]

        self.data = [raw_data[i] for i in split_indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        text: str = item["text"]
        positive_labels: List[str] = item["labels"]

        # Sample negatives and combine with positives
        negatives = negative_sampling(
            [positive_labels], self.all_labels, self.max_num_negatives
        )[0]

        candidate_labels = positive_labels + negatives
        # Shuffle so the model can't exploit position
        random.shuffle(candidate_labels)

        # Build binary target tensor
        pos_set = set(positive_labels)
        target = torch.zeros(self.max_num_labels, dtype=torch.float32)
        for j, lbl in enumerate(candidate_labels[: self.max_num_labels]):
            if lbl in pos_set:
                target[j] = 1.0

        return {
            "text": text,
            "labels": candidate_labels[: self.max_num_labels],
            "target": target,
        }


# ---------------------------------------------------------------------------
# Collate Function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collates a list of dataset samples into a batch dict.

    Returns:
        {
            'texts':   List[str]          — B input texts
            'labels':  List[List[str]]    — B candidate label lists
            'targets': Tensor [B, max_num_labels]
        }
    """
    texts = [item["text"] for item in batch]
    labels = [item["labels"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    return {"texts": texts, "labels": labels, "targets": targets}


def get_dataloader(
    data_path: str,
    max_num_labels: int = 5,
    max_num_negatives: int = 5,
    batch_size: int = 8,
    split: str = "train",
    train_split: float = 0.9,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, List[str]]:
    """
    Convenience function to build a DataLoader for the given split.

    Returns:
        (DataLoader, all_labels)
    """
    dataset = ZeroShotDataset(
        data_path=data_path,
        max_num_labels=max_num_labels,
        max_num_negatives=max_num_negatives,
        split=split,
        train_split=train_split,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.all_labels


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "data", "synthetic_data.json")
    loader, all_labels = get_dataloader(data_path, split="train", batch_size=4)
    print(f"All labels ({len(all_labels)}):", all_labels[:10], "...")
    batch = next(iter(loader))
    print("Texts:", batch["texts"])
    print("Labels:", batch["labels"])
    print("Targets shape:", batch["targets"].shape)
    print("Targets:", batch["targets"])
