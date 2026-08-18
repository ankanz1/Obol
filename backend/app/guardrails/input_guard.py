import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Patterns for input safety
PII_PATTERNS = [
    (r'\b\d{10,12}\b', 'phone_number'),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email'),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 'credit_card'),
    (r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b', 'ssn'),
]

INJECTION_PATTERNS = [
    r'ignore\s+previous\s+instructions',
    r'disregard\s+all\s+prior',
    r'system\s*:\s*you\s+are',
    r'assistant\s*:\s*you\s+are',
    r'<\|.*?\|>',  # Special tokens
    r'\[INST\].*?\[/INST\]',
    r'###\s*Instruction',
    r'```\s*system',
]

TOXICITY_KEYWORDS = [
    'hate', 'kill', 'murder', 'suicide', 'terrorist', 'bomb',
    'violence', 'abuse', 'harass', 'threat', 'racist', 'sexist'
]

MAX_INPUT_LENGTH = 2000


def check_input_safety(text: str) -> Tuple[bool, str]:
    """Check input for safety violations."""
    if not text or not text.strip():
        return False, "empty_input"
    
    text_lower = text.lower()
    
    # Check length
    if len(text) > MAX_INPUT_LENGTH:
        return False, "input_too_long"
    
    # Check PII
    for pattern, pii_type in PII_PATTERNS:
        if re.search(pattern, text):
            logger.warning(f"PII detected: {pii_type}")
            return False, f"pii_detected_{pii_type}"
    
    # Check injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Injection attempt detected: {pattern}")
            return False, "injection_attempt"
    
    # Check toxicity (basic keyword)
    for keyword in TOXICITY_KEYWORDS:
        if keyword in text_lower:
            logger.warning(f"Toxic keyword detected: {keyword}")
            return False, "toxic_content"
    
    return True, "safe"


def sanitize_input(text: str) -> str:
    """Sanitize input by removing/masking sensitive data."""
    # Mask PII
    for pattern, pii_type in PII_PATTERNS:
        text = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', text)
    
    # Remove injection patterns
    for pattern in INJECTION_PATTERNS:
        text = re.sub(pattern, '[REMOVED]', text, flags=re.IGNORECASE)
    
    return text