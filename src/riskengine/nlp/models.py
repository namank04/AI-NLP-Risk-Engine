"""Shared model loading: one device decision, cached pipelines, HF cache on D:."""

from __future__ import annotations

from functools import lru_cache

import riskengine.config  # noqa: F401  (sets HF_HOME before transformers is imported)


@lru_cache
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def batched_logits(model, tok, texts, pairs=None, batch_size: int = 32, max_length: int = 128,
                   truncation: bool | str = True):
    """Forward pass in batches with CUDA-OOM backoff (halve the batch and retry).

    The 6 GB laptop GPU is shared with the desktop, so headroom varies; a spike
    must slow inference down, never kill a replay.
    """
    import torch

    out, i, bs = [], 0, batch_size
    while i < len(texts):
        a = texts[i:i + bs]
        b = pairs[i:i + bs] if pairs is not None else None
        try:
            enc = tok(a, b, padding=True, truncation=truncation, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device()) for k, v in enc.items()}
            with torch.inference_mode():
                out.append(model(**enc).logits.float().cpu())
            i += len(a)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
    return torch.cat(out) if out else torch.zeros((0, model.config.num_labels))


@lru_cache
def load_classifier(model_name: str):
    """(tokenizer, model) in eval mode on the best device; fp16 on GPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device() == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype)
    model.to(device()).eval()
    return tok, model
