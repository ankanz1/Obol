#!/usr/bin/env python3
"""Download and convert embedding model to ONNX using transformers."""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from transformers import AutoTokenizer, AutoModel
import torch

def download_embedding_model():
    """Download multilingual-e5-small and export to ONNX."""
    model_id = "intfloat/multilingual-e5-small"
    output_dir = Path("backend/models/multilingual-e5-small")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_id}...")
    
    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    
    # Download model
    print("Downloading model...")
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    
    # Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = tokenizer(
        "Hello world",
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_dir / "model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"✅ Model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    download_embedding_model()