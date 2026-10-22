"""Load the assignment's embedding model on CPU and verify a real encoding."""

from importlib.metadata import version
import platform
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print(f"Python: {platform.python_version()}", flush=True)
    print(f"Python executable: {sys.executable}", flush=True)
    for package in ("sentence-transformers", "numpy", "torch", "transformers", "huggingface-hub"):
        print(f"{package}: {version(package)}", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print("Loading model on CPU...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    if model.device.type != "cpu":
        raise RuntimeError(f"Expected CPU, got {model.device}")
    for method in ("encode_document", "encode_query"):
        if not callable(getattr(model, method, None)):
            raise RuntimeError(f"Required method is missing: {method}")
    # This neutral sentence checks setup only; it is not assignment evidence.
    vector = model.encode_query(
        "Environment setup check.",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    norm = float(np.linalg.norm(vector))
    if vector.shape != (384,) or not np.isfinite(vector).all():
        raise RuntimeError(f"Invalid embedding: shape={vector.shape}")
    if not np.isclose(norm, 1.0, atol=1e-5):
        raise RuntimeError(f"Expected a normalized vector, got length {norm}")
    print(f"Device: {model.device}")
    print(f"Embedding shape: {vector.shape}")
    print(f"Embedding length: {norm:.8f}")
    print("encode_document / encode_query: available")
    print("PASS: model loaded and a real normalized 384-dimensional vector was produced on CPU.")


if __name__ == "__main__":
    main()
