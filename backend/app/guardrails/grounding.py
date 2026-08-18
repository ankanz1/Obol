import logging
import numpy as np
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Lazy load NLI model
_nli_session = None
_nli_tokenizer = None


def _load_nli_model():
    """Load DeBERTa-v3-small MNLI model via ONNX."""
    global _nli_session, _nli_tokenizer
    
    if _nli_session is not None:
        return
    
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        
        # Download ONNX model
        model_path = hf_hub_download(
            repo_id="onnx-community/deberta-v3-small",
            filename="model.onnx",
            subfolder="onnx"
        )
        
        providers = ['CPUExecutionProvider']
        try:
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
        except:
            pass
        
        _nli_session = ort.InferenceSession(model_path, providers=providers)
        _nli_tokenizer = AutoTokenizer.from_pretrained("onnx-community/deberta-v3-small")
        
        logger.info("NLI model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load NLI model: {e}")
        raise


def check_grounding(answer: str, context: str, threshold: float = 0.7) -> Tuple[bool, float]:
    """Check if answer is entailed by context using NLI."""
    if not answer or not context:
        return False, 0.0
    
    try:
        _load_nli_model()
        
        # Truncate if too long
        max_len = 512
        answer = answer[:max_len]
        context = context[:max_len * 2]
        
        # Format: premise = context, hypothesis = answer
        inputs = _nli_tokenizer(
            context,
            answer,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='np'
        )
        
        # Run inference
        outputs = _nli_session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )
        
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Labels: 0=entailment, 1=neutral, 2=contradiction (for MNLI)
        entailment_prob = float(probs[0])
        
        return entailment_prob >= threshold, entailment_prob
        
    except Exception as e:
        logger.warning(f"Grounding check failed, using fallback: {e}")
        # Fallback: simple keyword overlap
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return False, 0.0
        overlap = answer_words & context_words
        overlap_ratio = len(overlap) / len(answer_words)
        return overlap_ratio >= 0.3, overlap_ratio