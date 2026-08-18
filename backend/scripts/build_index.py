import polars as pl
from pathlib import Path
from typing import List, Dict, Any
import logging
import numpy as np
from tqdm import tqdm

from app.config import settings
from app.retrieval.embedder import get_embedder
from app.retrieval.qdrant_client import get_qdrant

logger = logging.getLogger(__name__)


def load_chunks(chunk_dir: Path, strategy: str = None) -> List[Dict[str, Any]]:
    """Load chunks from parquet files."""
    all_chunks = []
    
    if strategy:
        files = [chunk_dir / f"chunks_{strategy}.parquet"]
    else:
        files = list(chunk_dir.glob("chunks_*.parquet"))
    
    for file in files:
        if file.exists():
            df = pl.read_parquet(file)
            logger.info(f"Loaded {len(df)} chunks from {file}")
            all_chunks.extend(df.to_dicts())
        else:
            logger.warning(f"File not found: {file}")
    
    return all_chunks


def add_embeddings(chunks: List[Dict[str, Any]], embedder, batch_size: int = 64) -> List[Dict[str, Any]]:
    """Add embeddings to chunks."""
    texts = [chunk["content"] for chunk in chunks]
    
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedder.embed(texts, batch_size=batch_size)
    
    for i, chunk in enumerate(chunks):
        chunk["vector"] = embeddings[i]
    
    return chunks


def build_index(
    data_dir: Path = Path("backend/data/chunks/train"),
    strategy: str = None,
    batch_size: int = 100,
    embed_batch_size: int = 64,
    recreate: bool = False
):
    """Build Qdrant index from chunks."""
    logger.info("Starting index build...")
    
    # Initialize
    qdrant = get_qdrant()
    embedder = get_embedder()
    
    # Create collections
    qdrant.create_all_collections(recreate=recreate)
    
    # Load chunks
    all_chunks = load_chunks(data_dir, strategy)
    logger.info(f"Total chunks loaded: {len(all_chunks)}")
    
    # Group by language
    by_language = {}
    for chunk in all_chunks:
        lang = chunk.get("language", "en")
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append(chunk)
    
    # Process each language
    for lang, chunks in by_language.items():
        if lang not in settings.supported_languages:
            logger.warning(f"Skipping unsupported language: {lang}")
            continue
        
        logger.info(f"Processing {lang}: {len(chunks)} chunks")
        
        # Add embeddings
        chunks = add_embeddings(chunks, embedder, batch_size=embed_batch_size)
        
        # Upsert to Qdrant
        qdrant.upsert_chunks(chunks, lang, batch_size=batch_size)
        
        # Log stats
        stats = qdrant.get_collection_stats(lang)
        logger.info(f"  {lang} stats: {stats}")
    
    logger.info("Index build complete!")


def build_index_full_pipeline(
    raw_data_dir: Path = Path("backend/data/raw"),
    chunk_dir: Path = Path("backend/data/chunks"),
    strategy: str = None,
    recreate: bool = False
):
    """Run full pipeline: download -> chunk -> embed -> index."""
    from data.download import save_to_parquet, download_all_languages
    from data.chunking.pipeline import process_all_splits
    
    logger.info("=== FULL PIPELINE START ===")
    
    # Step 1: Download
    logger.info("Step 1: Downloading dataset...")
    train_path = raw_data_dir / "msmarco_xi_train.parquet"
    val_path = raw_data_dir / "msmarco_xi_val.parquet"
    
    save_to_parquet(download_all_languages("train"), train_path)
    save_to_parquet(download_all_languages("validation"), val_path)
    
    # Step 2: Chunk
    logger.info("Step 2: Chunking...")
    process_all_splits(raw_data_dir, chunk_dir)
    
    # Step 3: Build index
    logger.info("Step 3: Building index...")
    build_index(chunk_dir / "train", strategy=strategy, recreate=recreate)
    
    logger.info("=== FULL PIPELINE COMPLETE ===")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse args
    strategy = None
    recreate = False
    full = False
    
    for arg in sys.argv[1:]:
        if arg.startswith("--strategy="):
            strategy = arg.split("=")[1]
        elif arg == "--recreate":
            recreate = True
        elif arg == "--full":
            full = True
    
    if full:
        build_index_full_pipeline(recreate=recreate)
    else:
        build_index(strategy=strategy, recreate=recreate)