from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator
from dataclasses import dataclass
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    content: str
    language: str
    query_id: int
    query_type: str
    is_selected: bool
    source: str
    english_content: str
    metadata: Dict[str, Any]


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, row: Dict[str, Any], lang: str) -> List[Chunk]:
        pass


class PassageLevelChunker(BaseChunker):
    """Strategy 1: Each passage as separate chunk."""
    
    def chunk(self, row: Dict[str, Any], lang: str) -> List[Chunk]:
        chunks = []
        passages = row.get("passages", {})
        english_passages = passages.get("English_passages", [])
        translated_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        for i, (eng, trans, sel) in enumerate(zip(english_passages, translated_passages, is_selected)):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=trans,
                language=lang,
                query_id=row.get("query_id", 0),
                query_type=row.get("query_type", "UNKNOWN"),
                is_selected=bool(sel),
                source="passage_level",
                english_content=eng,
                metadata={
                    "passage_index": i,
                    "answer": row.get("Answer", ""),
                    "query": row.get("query", ""),
                }
            )
            chunks.append(chunk)
        
        return chunks


class QueryPassageChunker(BaseChunker):
    """Strategy 2: Combine query + selected passage."""
    
    def chunk(self, row: Dict[str, Any], lang: str) -> List[Chunk]:
        chunks = []
        passages = row.get("passages", {})
        english_passages = passages.get("English_passages", [])
        translated_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        query = row.get("query", "")
        
        for i, (eng, trans, sel) in enumerate(zip(english_passages, translated_passages, is_selected)):
            if sel:  # Only selected passages
                combined_content = f"Query: {query}\n\nPassage: {trans}"
                
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=combined_content,
                    language=lang,
                    query_id=row.get("query_id", 0),
                    query_type=row.get("query_type", "UNKNOWN"),
                    is_selected=True,
                    source="query_passage",
                    english_content=f"Query: {row.get('Eng_Query', '')}\n\nPassage: {eng}",
                    metadata={
                        "passage_index": i,
                        "answer": row.get("Answer", ""),
                        "original_query": query,
                    }
                )
                chunks.append(chunk)
        
        return chunks


class SlidingWindowChunker(BaseChunker):
    """Strategy 3: Sliding window for long passages."""
    
    def __init__(self, window_size: int = 512, overlap: int = 100):
        self.window_size = window_size
        self.overlap = overlap
    
    def _split_text(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.window_size:
            return [text]
        
        chunks = []
        step = self.window_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.window_size]
            chunks.append(" ".join(chunk_words))
            if i + self.window_size >= len(words):
                break
        return chunks
    
    def chunk(self, row: Dict[str, Any], lang: str) -> List[Chunk]:
        chunks = []
        passages = row.get("passages", {})
        english_passages = passages.get("English_passages", [])
        translated_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        for i, (eng, trans, sel) in enumerate(zip(english_passages, translated_passages, is_selected)):
            trans_windows = self._split_text(trans)
            eng_windows = self._split_text(eng)
            
            for j, (trans_win, eng_win) in enumerate(zip(trans_windows, eng_windows)):
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    content=trans_win,
                    language=lang,
                    query_id=row.get("query_id", 0),
                    query_type=row.get("query_type", "UNKNOWN"),
                    is_selected=bool(sel),
                    source="sliding_window",
                    english_content=eng_win,
                    metadata={
                        "passage_index": i,
                        "window_index": j,
                        "total_windows": len(trans_windows),
                        "answer": row.get("Answer", ""),
                        "query": row.get("query", ""),
                    }
                )
                chunks.append(chunk)
        
        return chunks


class LanguageRoutedChunker(BaseChunker):
    """Strategy 4: Language-specific collections (same as passage_level but with collection routing)."""
    
    def chunk(self, row: Dict[str, Any], lang: str) -> List[Chunk]:
        # Same chunking as passage_level, but will be routed to language-specific collection
        chunker = PassageLevelChunker()
        chunks = chunker.chunk(row, lang)
        for chunk in chunks:
            chunk.source = "language_routed"
            chunk.metadata["collection"] = f"msmarco_{lang}"
        return chunks


class MetadataEnrichedChunker(BaseChunker):
    """Strategy 5: Enriched metadata for filtering/reranking."""
    
    def chunk(self, row: Dict[str, Any], lang: str) -> List[Chunk]:
        chunks = []
        passages = row.get("passages", {})
        english_passages = passages.get("English_passages", [])
        translated_passages = passages.get("Translated_passages", [])
        is_selected = passages.get("is_selected", [])
        
        for i, (eng, trans, sel) in enumerate(zip(english_passages, translated_passages, is_selected)):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                content=trans,
                language=lang,
                query_id=row.get("query_id", 0),
                query_type=row.get("query_type", "UNKNOWN"),
                is_selected=bool(sel),
                source="metadata_enriched",
                english_content=eng,
                metadata={
                    "passage_index": i,
                    "answer": row.get("Answer", ""),
                    "query": row.get("query", ""),
                    "eng_query": row.get("Eng_Query", ""),
                    "eng_answer": row.get("Eng_Answer", ""),
                    "source_lang": row.get("source_lang", ""),
                    "target_lang": row.get("target_lang", ""),
                    "translation_model": row.get("meta", {}).get("model_name", ""),
                    "has_answer": bool(row.get("Answer", "")),
                    "answer_length": len(row.get("Answer", "")),
                    "passage_length": len(trans),
                }
            )
            chunks.append(chunk)
        
        return chunks


def get_all_chunkers() -> List[BaseChunker]:
    """Get all chunking strategies."""
    return [
        PassageLevelChunker(),
        QueryPassageChunker(),
        SlidingWindowChunker(),
        LanguageRoutedChunker(),
        MetadataEnrichedChunker(),
    ]


def chunk_row(row: Dict[str, Any], lang: str) -> List[Chunk]:
    """Apply all chunking strategies to a row."""
    all_chunks = []
    for chunker in get_all_chunkers():
        all_chunks.extend(chunker.chunk(row, lang))
    return all_chunks