import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# FastText disabled - using fallback script detection
# To enable: pip install fasttext-wheel (requires C++17 compiler)
_fasttext_available = False


def _simple_lang_detect(text: str) -> Tuple[str, float]:
    """Fallback simple language detection using character ranges."""
    if not text:
        return "en", 0.5
    
    # Count characters in different scripts
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    bengali = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    tamil = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    telugu = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    kannada = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')
    malayalam = sum(1 for c in text if '\u0D00' <= c <= '\u0D7F')
    gurmukhi = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
    oriya = sum(1 for c in text if '\u0B00' <= c <= '\u0B7F')
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    
    script_counts = {
        'hi': devanagari, 'mr': devanagari, 'ne': devanagari, 'sa': devanagari,
        'bn': bengali, 'as': bengali,
        'ta': tamil,
        'te': telugu,
        'gu': gujarati,
        'kn': kannada,
        'ml': malayalam,
        'pa': gurmukhi,
        'or': oriya,
        'ur': arabic, 'ks': arabic, 'sd': arabic,
    }
    
    # Find script with most characters
    max_script = max(script_counts.items(), key=lambda x: x[1])
    if max_script[1] > 0:
        return max_script[0], 0.8
    
    # Default to English for Latin script
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin > 0:
        return "en", 0.7
    
    return "en", 0.5


def detect_language(text: str) -> Tuple[str, float]:
    """Detect language of text using script detection (FastText disabled)."""
    if not text or not text.strip():
        return "en", 1.0
    
    # Use fallback script detection
    return _simple_lang_detect(text)


def is_supported_language(lang: str) -> bool:
    """Check if language is supported."""
    from app.config import settings
    return lang in settings.supported_languages