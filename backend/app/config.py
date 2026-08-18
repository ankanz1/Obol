from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # Sarvam AI
    sarvam_api_key: str = ""
    
    # Qdrant
    qdrant_url: str = "https://064705f0-4e5b-4c39-a888-e1517aa2219b.sa-east-1-0.aws.cloud.qdrant.io"
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NDQyMTk4OGEtMzk1Zi00YTM2LTljMDgtNTNmODkzZmJkYzdiIn0.Bq4iUSWSu_9BvC0A7sJI71_9ZGFd5T2L7Inpugzy_7Y"
    
    # Model settings
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "microsoft/deberta-v3-small"
    
    # Sarvam models
    stt_model: str = "saaras:v3-realtime"
    chat_model: str = "sarvam-105b"
    tts_model: str = "bulbul:v3"
    tts_voice: str = "aditya"
    
    # Pipeline settings
    retrieval_limit: int = 5
    retrieval_score_threshold: float = 0.65
    grounding_threshold: float = 0.7
    hallucination_threshold: float = 0.6
    max_conversation_turns: int = 10
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 100
    
    # Languages (18 Indic + English)
    supported_languages: List[str] = [
        "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa",
        "or", "as", "ur", "ne", "sa", "ks", "sd", "doi", "sat", "en"
    ]
    
    # Language codes mapping (HF dataset -> ISO)
    language_map: dict = {
        "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
        "mr": "mar_Deva", "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
        "pa": "pan_Guru", "or": "ory_Orya", "as": "asm_Beng", "ur": "urd_Arab",
        "ne": "npi_Deva", "sa": "san_Deva", "ks": "kas_Arab", "sd": "snd_Arab",
        "doi": "doi_Deva", "sat": "sat_Olck", "en": "eng_Latn"
    }
    
    # Refusal messages per language
    refusal_messages: dict = {
        "hi": "मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
        "bn": "এই প্রশ্নের উত্তর দেওয়ার জন্য আমার কাছে পর্যাপ্ত তথ্য নেই।",
        "ta": "இந்த கேள்விக்கு பதில் சொல்ல சரியான தகவல் என்னுடைய पासில் இல்லை।",
        "te": "ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి నా వెళ్ళుల minuto గురించి తగిన marges లేవు.",
        "mr": "हे प्रश्न उत्तर देण्यासाठी माझ्याकडे पुरेशी माहिती नाही.",
        "gu": "આ પ્રશ્નનો જવાબ આપવા મારી પાસે પર્યાપ્ત માહિતી નથી.",
        "kn": "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ಕೊಡುವನ Pause ನನ್ನ ಬಳಿ ಮಾಹಿತಿ ಇಲ್ಲ.",
        "ml": "ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ എനിക്ക് പര്യാപ്തമായ വിവരങ്ങളില്ല.",
        "pa": "ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਦੇਣ ਵਾਸਤੇ ਮੇਰੇ ਕੋਲ ਪਰਯਾਪਤ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
        "or": "ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ଦେବା ପାଇଁ ମୋର ପାଖରେ ପର୍ଯ୍ୟାପ୍ତ ତଥ୍ୟ ନାହିଁ।",
        "as": "এই প্ৰশ্নৰ উত্তৰ দিবলৈ মোৰ পাসে পৰ্যাপ্ত তথ্য নাই।",
        "ur": "اس سوال کا جواب دینے کے لیے میرے پاس کافی معلومات نہیں ہیں۔",
        "ne": "यस प्रश्नको उत्तर दिनका लागि ममै पर्याप्त जानकारी छैन।",
        "sa": "अस्य प्रश्नस्य उत्तरं दातुं मम सम्यक् ज्ञानं नास्ति।",
        "ks": "اس سوال کا جواب دینے کے لیے میرے پاس کافی معلومات نہیں ہیں۔",
        "sd": "هو سوال جو جواب ڏڻ واسطے منجهي معلومات نه مکمل آهين.",
        "doi": "ई प्रश्नको उत्तर दिनका लागि ममै पर्याप्त जानकारी छैन।",
        "sat": "ᱚᱞ ᱛᱩᱢᱮᱱ ᱠᱟᱱᱛᱤ ᱫᱤ ᱵᱟᱱᱞᱤ ᱚᱠᱚᱱ ᱵᱟᱨ ᱟᱹᱜᱤᱧ ᱟᱠᱤᱫ ᱵᱟᱱᱛᱤᱭ ᱚᱱᱚᱝᱜᱤᱭᱽᱽ",
        "en": "I don't have enough information to answer this question accurately."
    }
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10
    
    # Audio
    sample_rate: int = 16000
    audio_format: str = "pcm_16000"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()