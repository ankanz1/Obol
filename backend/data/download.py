from datasets import load_dataset
from typing import Iterator, Dict, Any, List
import polars as pl
from pathlib import Path
import logging
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from app.config import settings

logger = logging.getLogger(__name__)

# Map our language codes to dataset file prefixes
LANG_FILE_PREFIX = {
    "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
    "mr": "mar", "gu": "guj", "kn": "kan", "ml": "mal",
    "pa": "pan", "or": "ori", "as": "asm", "ur": "urd",
    "ne": "nep", "sa": "san", "ks": "kas", "sd": "snd",
    "doi": "doi", "sat": "sat"
}

# Reverse map for target_lang values in dataset
TARGET_LANG_MAP = {v: k for k, v in settings.language_map.items()}


def download_all_languages(split: str = "train", max_rows_total: int = None, max_rows_per_lang: int = None) -> Iterator[Dict]:
    """Download per-language parquet files and yield rows."""
    lang_codes = [c for c in settings.supported_languages if c != "en"]
    count = 0
    
    for lang_code in lang_codes:
        prefix = LANG_FILE_PREFIX.get(lang_code)
        if not prefix:
            logger.warning(f"No file prefix for {lang_code}, skipping")
            continue
        
        filename = f"{split}/{prefix}{split}.parquet"
        logger.info(f"Downloading {filename}...")
        
        try:
            local_path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=filename,
                repo_type="dataset"
            )
            
            df = pl.read_parquet(local_path)
            logger.info(f"  Loaded {len(df)} rows for {lang_code}")
            
            lang_count = 0
            for row in df.iter_rows(named=True):
                if max_rows_total and count >= max_rows_total:
                    return
                if max_rows_per_lang and lang_count >= max_rows_per_lang:
                    break
                
                row["_lang"] = lang_code
                yield row
                count += 1
                lang_count += 1
                
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            continue


def download_language(lang_code: str, split: str = "train") -> List[Dict]:
    """Download a single language's parquet file."""
    prefix = LANG_FILE_PREFIX.get(lang_code)
    if not prefix:
        logger.warning(f"No file prefix for {lang_code}")
        return []
    
    filename = f"{split}/{prefix}{split}.parquet"
    logger.info(f"Downloading {filename}...")
    
    try:
        local_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=filename,
            repo_type="dataset"
        )
        
        df = pl.read_parquet(local_path)
        logger.info(f"  Loaded {len(df)} rows for {lang_code}")
        
        rows = []
        for row in df.iter_rows(named=True):
            row["_lang"] = lang_code
            rows.append(row)
        return rows
        
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return []


def save_to_parquet(data_iter: Iterator[Dict], output_path: Path, batch_size: int = 10000):
    """Save streaming data to parquet in batches."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    batch = []
    total = 0
    
    for row in tqdm(data_iter, desc=f"Writing {output_path.name}"):
        batch.append(row)
        if len(batch) >= batch_size:
            df = pl.DataFrame(batch)
            if output_path.exists():
                df.write_parquet(output_path, append=True)
            else:
                df.write_parquet(output_path)
            total += len(batch)
            batch = []
    
    if batch:
        df = pl.DataFrame(batch)
        if output_path.exists():
            df.write_parquet(output_path, append=True)
        else:
            df.write_parquet(output_path)
        total += len(batch)
    
    logger.info(f"Saved {total} rows to {output_path}")
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path("backend/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download train split - limit to 1000 rows per language for testing
    train_path = data_dir / "msmarco_xi_train.parquet"
    save_to_parquet(
        download_all_languages("train", max_rows_total=18000, max_rows_per_lang=1000),  # ~18K total, 1K per lang
        train_path
    )
    
    # Download validation split
    val_path = data_dir / "msmarco_xi_val.parquet"
    save_to_parquet(
        download_all_languages("validation", max_rows_total=1800, max_rows_per_lang=100),
        val_path
    )