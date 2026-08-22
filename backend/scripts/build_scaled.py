#!/usr/bin/env python3
"""Scaled-down MSMARCO-XI index build.

Processes each available language one at a time:
  download full train parquet -> stream first N rows -> chunk (5 strategies)
  -> ONNX embed -> upsert to Qdrant `msmarco_{lang}` -> delete raw file.

Only languages with train parquet files are processed
(as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, ur). ks/sd/doi/sat have no
train files and are left empty. An `en` collection is synthesized from the
English side of the `hi` rows.

Designed for low-RAM boxes: rows are streamed with pyarrow iter_batches and
each 3.7GB raw file is deleted right after the subset is written.
"""

import argparse
import gc
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

# Xet transfers are unstable on low-RAM hosts (background writer channel
# closes mid-download). Disable them: fall back to the resumable HTTP path.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import polars as pl

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from huggingface_hub import hf_hub_download  # noqa: E402

from data.chunking.strategies import Chunk, get_all_chunkers  # noqa: E402
from app.retrieval.embedder import get_embedder  # noqa: E402
from app.retrieval.qdrant_client import get_qdrant  # noqa: E402

logger = logging.getLogger("build_scaled")

# Language -> parquet file prefix (only languages that have train files)
LANG_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "ur": "urd",
}

HF_TMP_DIR = BACKEND_DIR / ".hf_tmp"
SUBSET_DIR = BACKEND_DIR / "data" / "raw_subset"

DEFAULT_ROWS_PER_LANG = 500

# Embed+upsert slice size: bounds peak RAM (20K chunks held in memory at once
# was the fragile point on a 3.7GB box) and yields frequent progress lines.
EMBED_SLICE = 1000
# Empirical chunks-per-row from the 500-row subsets (~20319/500).
CHUNKS_PER_ROW = 40


def chunk_to_dict(c: Chunk) -> Dict:
    return {
        "id": c.id,
        "content": c.content,
        "language": c.language,
        "query_id": c.query_id,
        "query_type": c.query_type,
        "is_selected": c.is_selected,
        "source": c.source,
        "english_content": c.english_content,
        **c.metadata,
    }


def download_train_parquet(lang: str, prefix: str, max_attempts: int = 4) -> str:
    """Download the language's full train parquet; returns local path.

    Retries on failure; partial downloads in the cache are resumed on retry.
    """
    HF_TMP_DIR.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"[{lang}] Downloading train/{prefix}train.parquet "
                        f"(attempt {attempt}/{max_attempts}) ...")
            path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=f"train/{prefix}train.parquet",
                repo_type="dataset",
                cache_dir=str(HF_TMP_DIR),
            )
            logger.info(f"[{lang}] Download complete: {path}")
            return path
        except Exception as e:
            last_err = e
            logger.warning(f"[{lang}] download attempt {attempt} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"download failed after {max_attempts} attempts: {last_err}")


def subset_first_rows(parquet_path: str, n: int) -> List[Dict]:
    """Stream first `n` rows from the parquet into a list of dicts.

    Uses polars streaming so only the pages needed for the head are decoded,
    avoiding the multi-GB peak that `pyarrow.iter_batches` hit on these
    single-row-group files.
    """
    t0 = time.time()
    df = (
        pl.scan_parquet(parquet_path)
        .head(n)
        .collect(engine="streaming")
    )
    logger.info(f"[subset] took {df.height} rows in {time.time()-t0:.1f}s")
    return df.to_dicts()


def free_download_cache():
    shutil.rmtree(HF_TMP_DIR, ignore_errors=True)
    gc.collect()


def save_subset(rows: List[Dict], lang: str):
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSET_DIR / f"{lang}.parquet"
    pl.DataFrame(rows).write_parquet(out)
    logger.info(f"[{lang}] subset written: {out} ({len(rows)} rows)")


def chunk_rows(rows: List[Dict], lang: str) -> List[Dict]:
    chunkers = get_all_chunkers()
    chunks: List[Dict] = []
    for row in rows:
        for chunker in chunkers:
            for chunk in chunker.chunk(row, lang):
                chunks.append(chunk_to_dict(chunk))
    return chunks


def load_or_download_subset(lang: str, prefix: str, rows_per_lang: int) -> List[Dict]:
    """Reuse a cached subset parquet if it exists and matches --rows-per-lang.

    Returns the rows for the language. When a matching subset already exists the
    multi-minute download+subsample is skipped entirely.
    """
    subset_path = SUBSET_DIR / f"{lang}.parquet"
    if subset_path.exists():
        df = pl.read_parquet(subset_path)
        if df.height == rows_per_lang:
            logger.info(f"[{lang}] reusing cached subset {subset_path} "
                        f"({df.height} rows)")
            return df.to_dicts()
        logger.info(f"[{lang}] cached subset has {df.height} rows, expected "
                    f"{rows_per_lang}; re-downloading")
    path = download_train_parquet(lang, prefix)
    rows = subset_first_rows(path, rows_per_lang)
    free_download_cache()
    save_subset(rows, lang)
    return rows


def embed_and_upsert(chunks: List[Dict], lang: str, batch_size: int = 32,
                    slice_size: int = EMBED_SLICE):
    if not chunks:
        logger.warning(f"[{lang}] no chunks to index")
        return 0
    logger.info(f"[{lang}] embedding {len(chunks)} chunks ...")
    embedder = get_embedder()
    qdrant = get_qdrant()
    done = 0
    for i in range(0, len(chunks), slice_size):
        slice_ = chunks[i:i + slice_size]
        texts = [c["content"] for c in slice_]
        vectors = embedder.embed(texts, batch_size=batch_size)
        for c, v in zip(slice_, vectors):
            c["vector"] = v
        qdrant.upsert_chunks(slice_, lang, batch_size=100)
        done += len(slice_)
        logger.info(f"[{lang}] upserted {done}/{len(chunks)} chunks")
        del slice_, texts, vectors
        gc.collect()
    stats = qdrant.get_collection_stats(lang)
    logger.info(f"[{lang}] indexed. stats={stats}")
    return len(chunks)


def already_indexed(lang: str, min_points: int = 20000) -> bool:
    """True only if the collection already holds >= min_points points.

    Uses a full-count threshold so a partially-built collection (e.g. after an
    OOM kill mid-language) is NOT treated as done and gets rebuilt from the
    cached subset.
    """
    try:
        stats = get_qdrant().get_collection_stats(lang)
        pts = stats.get("points_count")
        return isinstance(pts, int) and pts >= min_points
    except Exception:
        return False


def to_english_row(row: Dict) -> Dict:
    passages = row.get("passages", {})
    eng = passages.get("English_passages", [])
    sel = passages.get("is_selected", [])
    return {
        "query": row.get("Eng_Query") or row.get("query", ""),
        "Answer": row.get("Eng_Answer") or row.get("Answer", ""),
        "query_id": row.get("query_id", 0),
        "query_type": row.get("query_type", "UNKNOWN"),
        "passages": {
            "English_passages": eng,
            "Translated_passages": eng,
            "is_selected": sel,
        },
        "Eng_Query": row.get("Eng_Query", ""),
        "Eng_Answer": row.get("Eng_Answer", ""),
        "source_lang": row.get("source_lang", ""),
        "target_lang": row.get("target_lang", ""),
        "meta": row.get("meta", {}),
    }


def main():
    ap = argparse.ArgumentParser(description="Scaled MSMARCO-XI index build")
    ap.add_argument("--rows-per-lang", type=int, default=DEFAULT_ROWS_PER_LANG)
    ap.add_argument("--langs", type=str, default=",".join(LANG_PREFIX.keys()))
    ap.add_argument("--recreate", action="store_true",
                    help="recreate Qdrant collections first")
    ap.add_argument("--no-en", action="store_true",
                    help="skip the synthesized English collection")
    ap.add_argument("--embed-slice", type=int, default=EMBED_SLICE,
                    help="chunks embedded+upserted per pass (RAM bound)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="ONNX embed batch size (RAM bound)")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap ONNX/CPU threads (reduces peak RAM)")
    args = ap.parse_args()

    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    missing = [l for l in langs if l not in LANG_PREFIX]
    if missing:
        logger.warning(f"Skipping languages without train files: {missing}")
    langs = [l for l in langs if l in LANG_PREFIX]

    qdrant = get_qdrant()
    logger.info("Creating collections ...")
    qdrant.create_all_collections(recreate=args.recreate)

    en_rows: List[Dict] = []
    total_points = 0
    start_all = time.time()
    expected_min = args.rows_per_lang * CHUNKS_PER_ROW

    for lang in langs:
        t0 = time.time()
        if not args.recreate and already_indexed(lang, expected_min):
            logger.info(f"[{lang}] already indexed, skipping")
            if lang == "hi":
                rows = load_or_download_subset(
                    lang, LANG_PREFIX[lang], args.rows_per_lang)
                en_rows.extend(rows)
                del rows
                gc.collect()
            continue
        try:
            rows = load_or_download_subset(lang, LANG_PREFIX[lang], args.rows_per_lang)
            if lang == "hi":
                en_rows.extend(rows)
            n = 0
            # Process in row-batches so the full 20K-chunk dict list is never
            # held in memory at once (the OOM trigger on this 3.7GB box).
            row_batch = max(1, args.embed_slice // CHUNKS_PER_ROW)
            for i in range(0, len(rows), row_batch):
                batch_rows = rows[i:i + row_batch]
                chunks = chunk_rows(batch_rows, lang)
                del batch_rows
                gc.collect()
                n += embed_and_upsert(chunks, lang, args.batch_size, args.embed_slice)
                del chunks
                gc.collect()
            total_points += n
            logger.info(f"[{lang}] === DONE in {time.time()-t0:.0f}s "
                        f"({n} points) ===")
        except Exception as e:
            # Keep the cache so partial downloads resume on a future run.
            logger.exception(f"[{lang}] FAILED: {e}")
            continue

    if not args.no_en:
        t0 = time.time()
        if args.recreate or not already_indexed("en", expected_min):
            if not en_rows:
                hi_path = SUBSET_DIR / "hi.parquet"
                if hi_path.exists():
                    logger.info("[en] loading cached hi subset for English rows")
                    en_rows = pl.read_parquet(hi_path).to_dicts()
            if en_rows:
                logger.info(f"[en] building from {len(en_rows)} hi rows ...")
                en_all = [to_english_row(r) for r in en_rows]
                del en_rows
                gc.collect()
                n = 0
                row_batch = max(1, args.embed_slice // CHUNKS_PER_ROW)
                for i in range(0, len(en_all), row_batch):
                    batch_rows = en_all[i:i + row_batch]
                    chunks = chunk_rows(batch_rows, "en")
                    del batch_rows
                    gc.collect()
                    n += embed_and_upsert(chunks, "en", args.batch_size,
                                          args.embed_slice)
                    del chunks
                    gc.collect()
                total_points += n
                logger.info(f"[en] === DONE in {time.time()-t0:.0f}s "
                            f"({n} points) ===")
            else:
                logger.warning("[en] no hi rows available, skipping")
        else:
            logger.info("[en] already indexed, skipping")

    logger.info(f"=== BUILD COMPLETE in {time.time()-start_all:.0f}s "
                f"({total_points} points total) ===")


if __name__ == "__main__":
    main()