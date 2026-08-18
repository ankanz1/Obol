import polars as pl
from pathlib import Path
from typing import Iterator, List
import logging
from tqdm import tqdm

from app.config import settings
from .strategies import chunk_row, Chunk, get_all_chunkers

logger = logging.getLogger(__name__)


def process_parquet_file(input_path: Path, output_dir: Path, batch_size: int = 1000):
    """Process parquet file and apply all chunking strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize output files for each strategy
    chunkers = get_all_chunkers()
    strategy_names = [c.__class__.__name__.replace("Chunker", "").lower() for c in chunkers]
    
    output_files = {
        name: output_dir / f"chunks_{name}.parquet"
        for name in strategy_names
    }
    
    # Buffers for batch writing
    buffers = {name: [] for name in strategy_names}
    
    def flush_buffer(name: str):
        if buffers[name]:
            df = pl.DataFrame(buffers[name])
            if output_files[name].exists():
                df.write_parquet(output_files[name], append=True)
            else:
                df.write_parquet(output_files[name])
            buffers[name].clear()
    
    logger.info(f"Processing {input_path}...")
    df = pl.read_parquet(input_path)
    
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Chunking"):
        lang = row.get("_lang", "en")
        
        # Apply each chunking strategy
        for chunker in chunkers:
            strategy_name = chunker.__class__.__name__.replace("Chunker", "").lower()
            chunks = chunker.chunk(row, lang)
            
            for chunk in chunks:
                buffers[strategy_name].append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "language": chunk.language,
                    "query_id": chunk.query_id,
                    "query_type": chunk.query_type,
                    "is_selected": chunk.is_selected,
                    "source": chunk.source,
                    "english_content": chunk.english_content,
                    **chunk.metadata
                })
        
        # Flush periodically
        if len(buffers[strategy_names[0]]) >= batch_size:
            for name in strategy_names:
                flush_buffer(name)
    
    # Final flush
    for name in strategy_names:
        flush_buffer(name)
    
    # Log stats
    for name in strategy_names:
        count = pl.read_parquet(output_files[name]).height
        logger.info(f"  {name}: {count} chunks -> {output_files[name]}")
    
    return output_files


def process_all_splits(data_dir: Path = Path("backend/data/raw"), output_dir: Path = Path("backend/data/chunks")):
    """Process train and validation splits."""
    for split in ["train", "validation"]:
        input_path = data_dir / f"msmarco_xi_{split}.parquet"
        if input_path.exists():
            split_output = output_dir / split
            process_parquet_file(input_path, split_output)
        else:
            logger.warning(f"Split file not found: {input_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_all_splits()