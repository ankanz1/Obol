import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Lazy load FastText
_fasttext_model = None
_fasttext_available = None


def _load_fasttext():
    """Load FastText language detection model."""
    global _fasttext_model, _fasttext_available
    
    if _fasttext_model is not None:
        return
    
    if _fasttext_available is False:
        return
    
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        
        # Download lid.176.bin
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin"
        )
        
        _fasttext_model = fasttext.load_model(model_path)
        _fasttext_available = True
        logger.info("FastText language detection model loaded")
        
    except Exception as e:
        _fasttext_available = False
        logger.warning(f"FastText not available: {e}")


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
    """Detect language of text using FastText (with fallback)."""
    if not text or not text.strip():
        return "en", 1.0
    
    try:
        _load_fasttext()
        
        if _fasttext_available and _fasttext_model:
            # FastText expects single line
            text = text.replace('\n', ' ')[:500]
            
            predictions = _fasttext_model.predict(text, k=1)
            lang_code = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])
            
            # Map FastText codes to our codes
            lang_map = {
                'hi': 'hi', 'bn': 'bn', 'ta': 'ta', 'te': 'te',
                'mr': 'mr', 'gu': 'gu', 'kn': 'kn', 'ml': 'ml',
                'pa': 'pa', 'or': 'or', 'as': 'as', 'ur': 'ur',
                'ne': 'ne', 'sa': 'sa', 'ks': 'ks', 'sd': 'sd',
                'doi': 'doi', 'sat': 'sat', 'en': 'en',
                'eng': 'en', 'hin': 'hi', 'ben': 'bn', 'tam': 'ta',
                'tel': 'te', 'mar': 'mr', 'guj': 'gu', 'kan': 'kn',
                'mal': 'ml', 'pan': 'pa', 'ori': 'or', 'asm': 'as',
                'urd': 'ur', 'nep': 'ne', 'san': 'sa', 'kas': 'ks',
                'snd': 'sd', 'sat': 'sat'
            }
            
            mapped_lang = lang_map.get(lang_code, 'en')
            return mapped_lang, confidence
            
    except Exception as e:
        logger.warning(f"FastText detection failed, using fallback: {e}")
    
    # Fallback to simple script detection
    return _simple_lang_detect(text)


def is_supported_language(lang: str) -> bool:
    """Check if language is supported."""
    from app.config import settings
    return lang in settings.supported_languages


def is_supported_language(lang: str) -> bool:
    """Check if language is supported."""
    from app.config import settings
    return lang in settings.supported_languages