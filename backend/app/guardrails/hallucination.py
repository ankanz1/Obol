import logging
from typing import Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def check_hallucination(answer: str, context: str, threshold: float = 0.3) -> Tuple[bool, float]:
    """Self-consistency hallucination check via semantic similarity."""
    if not answer or not context:
        return False, 0.0
    
    try:
        # Normalize
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if not answer_words:
            return False, 0.0
        
        # Word overlap ratio
        overlap = answer_words & context_words
        overlap_ratio = len(overlap) / len(answer_words)
        
        # Sequence similarity for phrase-level check
        seq_ratio = SequenceMatcher(None, answer.lower(), context.lower()).ratio()
        
        # Combined score
        score = (overlap_ratio + seq_ratio) / 2
        
        return score >= threshold, score
        
    except Exception as e:
        logger.error(f"Hallucination check failed: {e}")
        return True, 0.5  # Fail open


def check_self_consistency(
    answer1: str,
    answer2: str,
    threshold: float = 0.6
) -> Tuple[bool, float]:
    """Check consistency between two generations."""
    if not answer1 or not answer2:
        return False, 0.0
    
    try:
        ratio = SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()
        return ratio >= threshold, ratio
    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        return True, 0.5