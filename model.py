"""
BiEncoder Model for Zero-Shot Text Classification.

This module implements a BiEncoder architecture where a shared transformer
encoder encodes both texts and labels, then similarity scores are computed
via dot product to enable zero-shot classification.
"""

import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig


class BiEncoderModel(nn.Module):
    """
    BiEncoder model for zero-shot text classification.

    Uses a shared transformer encoder (default: bert-base-uncased) to encode
    both input texts and candidate labels independently. Classification is
    performed by computing cosine similarity between text and label embeddings.

    Architecture:
        - Shared transformer encoder for texts and labels
        - Mask-aware mean pooling to get fixed-size embeddings
        - Scaled dot-product similarity for classification scores
        - Sigmoid cross-entropy loss for multi-label training

    Args:
        model_name (str): Hugging Face model identifier.
        max_num_labels (int): Maximum number of candidate labels per sample.
        temperature (float): Scaling factor for similarity scores. Default: 1.0.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_num_labels: int = 5,
        temperature: float = 1.0,
    ):
        super(BiEncoderModel, self).__init__()
        self.model_name = model_name
        self.max_num_labels = max_num_labels
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Shared encoder for both text and labels
        self.shared_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Optional projection head to reduce dimensionality
        hidden_size = self.shared_encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def encode(self, texts_or_labels: List[str]) -> torch.Tensor:
        """
        Encodes a list of texts or labels using the shared encoder.

        Args:
            texts_or_labels: List of strings to encode.

        Returns:
            Tensor of shape [N, D] containing L2-normalized embeddings.
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            texts_or_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.shared_encoder(**inputs)

        # Mask-aware mean pooling: last_hidden_state [B, seq_len, D]
        att_mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * att_mask).sum(1) / att_mask.sum(1)

        # Project and L2-normalize
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        texts: List[str],
        batch_labels: List[List[str]],
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass computing similarity scores between texts and their candidate labels.

        Args:
            texts: List of B input texts.
            batch_labels: List of B lists, each containing candidate labels for that text.
            targets: Optional tensor of shape [B, max_num_labels] with binary targets (0/1).

        Returns:
            If targets is None:
                scores (Tensor): [B, max_num_labels] — sigmoid similarity scores.
                mask (Tensor): [B, max_num_labels] — boolean mask for valid label positions.
            If targets is provided:
                loss (Tensor): Scalar sigmoid cross-entropy loss.
                scores (Tensor): [B, max_num_labels] — sigmoid similarity scores.
                mask (Tensor): [B, max_num_labels] — boolean mask for valid label positions.
        """
        B = len(texts)
        device = next(self.parameters()).device

        # Flatten all labels in the batch and encode together (efficiency)
        all_labels = [label for labels in batch_labels for label in labels]
        label_embeddings = self.encode(all_labels)  # [total_labels, D]

        # Encode texts
        text_embeddings = self.encode(texts)  # [B, D]

        # Reconstruct padded label tensor [B, max_num_labels, D]
        label_counts = [len(labels) for labels in batch_labels]
        D = label_embeddings.size(-1)
        padded_label_embeddings = torch.zeros(B, self.max_num_labels, D, device=device)
        mask = torch.zeros(B, self.max_num_labels, dtype=torch.bool, device=device)

        current = 0
        for i, count in enumerate(label_counts):
            if count > 0:
                end = current + count
                actual_count = min(count, self.max_num_labels)
                padded_label_embeddings[i, :actual_count, :] = label_embeddings[current : current + actual_count]
                mask[i, :actual_count] = True
                current = end

        # Cosine similarity scaled by learned temperature
        # text_embeddings: [B, D] → [B, 1, D]
        # padded_label_embeddings: [B, max_num_labels, D]
        scores = torch.bmm(
            padded_label_embeddings, text_embeddings.unsqueeze(2)
        ).squeeze(2)  # [B, max_num_labels]
        scores = scores * self.temperature.abs()
        scores = torch.sigmoid(scores)

        if targets is not None:
            targets = targets.to(device).float()
            # Only compute loss on valid (non-padding) positions
            loss = F.binary_cross_entropy(scores * mask.float(), targets * mask.float(), reduction="sum")
            loss = loss / mask.float().sum().clamp(min=1.0)
            return loss, scores, mask

        return scores, mask

    @torch.no_grad()
    def forward_predict(
        self, texts: List[str], labels: List[List[str]], threshold: float = 0.5
    ):
        """
        Zero-shot prediction returning label scores for each text.

        Args:
            texts: List of input texts.
            labels: List of candidate label lists for each text.
            threshold: Confidence threshold for accepting a label as positive.

        Returns:
            List of dicts with 'text', 'scores', and 'predicted_labels' keys.
        """
        self.eval()
        scores, mask = self.forward(texts, labels)
        results = []
        for i, text in enumerate(texts):
            text_result = {}
            for j, label in enumerate(labels[i]):
                if mask[i, j]:
                    text_result[label] = float(f"{scores[i, j].item():.4f}")
            predicted = [lbl for lbl, sc in text_result.items() if sc >= threshold]
            results.append(
                {
                    "text": text,
                    "scores": text_result,
                    "predicted_labels": predicted,
                }
            )
        return results

    def save_pretrained(self, save_directory: str):
        """
        Save model weights, tokenizer, and config to a directory.

        Args:
            save_directory: Path to the output directory.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save transformer encoder + tokenizer
        self.shared_encoder.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save projection head and temperature
        torch.save(
            {
                "projection": self.projection.state_dict(),
                "temperature": self.temperature.data,
            },
            os.path.join(save_directory, "bi_encoder_head.pt"),
        )

        # Save custom config
        custom_config = {
            "model_name": self.model_name,
            "max_num_labels": self.max_num_labels,
            "temperature": float(self.temperature.data),
            "architectures": ["BiEncoderModel"],
        }
        with open(os.path.join(save_directory, "bi_encoder_config.json"), "w") as f:
            json.dump(custom_config, f, indent=2)

        print(f"✅ Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_directory: str) -> "BiEncoderModel":
        """
        Load a BiEncoderModel from a saved directory or Hugging Face Hub.

        Args:
            model_directory: Path or Hugging Face Hub model ID.

        Returns:
            Loaded BiEncoderModel instance (eval mode).
        """
        config_path = os.path.join(model_directory, "bi_encoder_config.json")

        if os.path.exists(config_path):
            with open(config_path) as f:
                custom_config = json.load(f)
        else:
            # Try loading from HF Hub – fall back to defaults
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_directory, filename="bi_encoder_config.json")
            with open(config_path) as f:
                custom_config = json.load(f)

        model = cls(
            model_name=model_directory,  # HF will load weights from here
            max_num_labels=custom_config["max_num_labels"],
            temperature=custom_config.get("temperature", 1.0),
        )

        head_path = os.path.join(model_directory, "bi_encoder_head.pt")
        if not os.path.exists(head_path):
            from huggingface_hub import hf_hub_download
            head_path = hf_hub_download(repo_id=model_directory, filename="bi_encoder_head.pt")

        head_state = torch.load(head_path, map_location="cpu")
        model.projection.load_state_dict(head_state["projection"])
        model.temperature.data = head_state["temperature"]

        model.eval()
        print(f"✅ Model loaded from {model_directory}")
        return model

    def push_to_hub(self, repo_id: str, private: bool = True, token: Optional[str] = None):
        """
        Upload model to the Hugging Face Hub.

        Args:
            repo_id: Hub repository ID, e.g. 'username/bi-encoder-model'.
            private: Whether to create a private repository.
            token: Hugging Face API token. Uses cached token if None.
        """
        import tempfile
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                repo_type="model",
            )
        print(f"✅ Model pushed to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    max_num_labels = 5
    model = BiEncoderModel(model_name, max_num_labels)

    texts = ["I love machine learning.", "Deep learning models are powerful."]
    batch_labels = [
        ["AI", "Machine Learning"],
        ["Deep Learning", "Neural Networks", "AI"],
    ]

    # Forward pass
    scores, mask = model(texts, batch_labels)
    print("Scores:", scores)
    print("Mask:", mask)

    # Prediction with JSON output
    predictions = model.forward_predict(texts, batch_labels)
    print("Predictions:", predictions)
