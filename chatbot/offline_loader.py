# offline_loader.py  (import this file in utils.py)

import os
import torch
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
# Absolute path to the model folder you copied
EMBED_MODEL_DIR = Path(__file__).resolve().parent / "chatbot" / "models" / "all-MiniLM-L6-v2"

# ✅ NEW (correct since you're now inside `chatbot/`):
EMBED_MODEL_DIR = Path(__file__).resolve().parent / "models" / "all-MiniLM-L6-v2"

def _ensure_files_exist() -> None:
    """Quick sanity check – raise immediately if any critical file is missing."""
    required = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
    missing = [f for f in required if not (EMBED_MODEL_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"❌ Missing files {missing} in {EMBED_MODEL_DIR}. "
            "Copy the model locally before starting the server."
        )

def load_sentence_transformer(offload: bool = True) -> SentenceTransformer:
    """
    Load Sentence‑BERT completely offline and quickly.

    Args:
        offload (bool): If True, puts the model on CPU with half precision
                        to minimise RAM and speed up load.

    Returns:
        SentenceTransformer
    """
    _ensure_files_exist()

    # Force Hugging Face 'offline' mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Load
    model = SentenceTransformer(str(EMBED_MODEL_DIR))

    if offload:
        # speed: cast weights to float16 and keep on CPU
        model.to(torch.float16)
        model.eval()

    return model


class OfflineSentenceTransformerEmbeddings(SentenceTransformerEmbeddings):
    """
    A drop‑in replacement that always loads from the local folder above.
    Usage:
        model = OfflineSentenceTransformerEmbeddings()
    """
    def __init__(self, **kwargs):
        super().__init__(
            model_name=str(EMBED_MODEL_DIR),
            model_kwargs={
                "device": "cpu",          # keep CPU‑only – no CUDA init delay
                "trust_remote_code": False,
                "local_files_only": True, # extra safety
            },
            **kwargs,
        )
