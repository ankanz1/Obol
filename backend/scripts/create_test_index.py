#!/usr/bin/env python3
"""Create test data and build a quick index for development."""

import json
import polars as pl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.chunking.strategies import chunk_row
from app.retrieval.embedder import get_embedder
from app.retrieval.qdrant_client import get_qdrant

def create_test_data():
    """Create test data from sample JSONL."""
    data_dir = Path("backend/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    test_data = []
    with open("data/sample_data.jsonl") as f:
        for line in f:
            row = json.loads(line)
            row["_lang"] = "hi"
            test_data.append(row)
    
    # Also create some English data
    en_data = {
        "query": "what is the capital of india",
        "Answer": "new delhi is the capital of india",
        "query_id": 2,
        "query_type": "ENTITY",
        "passages": {
            "is_selected": [1, 0, 0],
            "English_passages": ["New Delhi is the capital of India.", "Mumbai is the financial capital."],
            "Translated_passages": ["New Delhi is the capital of India.", "Mumbai is the financial capital."]
        },
        "Eng_Query": "what is the capital of india",
        "Eng_Answer": "new delhi is the capital of india",
        "source_lang": "eng_Latn",
        "target_lang": "eng_Latn",
        "meta": {"model_name": "test"}
    }
    en_data["_lang"] = "en"
    test_data.append(en_data)
    
    df = pl.DataFrame(test_data)
    train_path = data_dir / "msmarco_xi_train.parquet"
    df.write_parquet(train_path)
    print(f"Created test data: {len(test_data)} rows -> {train_path}")
    
    # Validation
    val_path = data_dir / "msmarco_xi_val.parquet"
    df.write_parquet(val_path)
    print(f"Created validation data -> {val_path}")


def build_test_index():
    """Build a quick test index."""
    print("Building test index...")
    
    # Chunk the test data
    from data.chunking.pipeline import process_parquet_file
    data_dir = Path("backend/data/raw")
    chunk_dir = Path("backend/data/chunks/train")
    process_parquet_file(data_dir / "msmarco_xi_train.parquet", chunk_dir)
    
    # Build index
    from scripts.build_index import build_index
    build_index(chunk_dir, strategy=None, batch_size=50, embed_batch_size=32, recreate=True)
    
    print("Test index built!")


if __name__ == "__main__":
    create_test_data()
    build_test_index()