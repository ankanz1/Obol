import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# Output safety patterns
TOXICITY_PATTERNS = [
    r'\b(hate|kill|murder|suicide|terrorist|bomb)\b',
    r'\b(violence|abuse|harass|threat)\b',
    r'\b(racist|sexist|discriminat)\w*\b',
]

PII_PATTERNS = [
    (r'\b\d{10,12}\b', 'phone_number'),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 'credit_card'),
]


def check_output_safety(text: str) -> Tuple[bool, str]:
    """Check generated output for safety violations."""
    if not text or not text.strip():
        return False, "empty_output"
    
    text_lower = text.lower()
    
    # Check toxicity
    for pattern in TOXICITY_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Toxic output detected: {pattern}")
            return False, "toxic_output"
    
    # Check PII leakage
    for pattern, pii_type in PII_PATTERNS:
        if re.search(pattern, text):
            logger.warning(f"PII in output: {pii_type}")
            return False, f"pii_leakage_{pii_type}"
    
    return True, "safe"


def sanitize_output(text: str) -> str:
    """Sanitize output by masking PII."""
    for pattern, pii_type in PII_PATTERNS:
        text = re.sub(pattern, f'[REDACTED]', text)
    return text