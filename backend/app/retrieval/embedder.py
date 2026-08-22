import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Union
import logging
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ONNXEmbedder:
    """Fast embedder for multilingual-e5-small using ONNX runtime."""
    
    def __init__(self, model_path: str = None, providers: List[str] = None):
        # Use local model if available (resolve relative to this file, cwd-independent)
        if model_path is None:
            local_path = Path(__file__).resolve().parent.parent.parent / "models" / "multilingual-e5-small"
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = "intfloat/multilingual-e5-small"
        
        self.model_path = model_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load ONNX model
        if providers is None:
            providers = ['CPUExecutionProvider']
            try:
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
            except:
                pass
        
        # Try to load local ONNX model, otherwise download from HF
        onnx_path = None
        if Path(self.model_path).exists():
            onnx_path = Path(self.model_path) / "model.onnx"
            if not onnx_path.exists():
                onnx_path = Path(self.model_path) / "onnx" / "model.onnx"
        
        if onnx_path is None or not onnx_path.exists():
            # Download ONNX model from HF
            try:
                onnx_path = hf_hub_download(
                    repo_id="onnx-community/multilingual-e5-small",
                    filename="model.onnx",
                    subfolder="onnx"
                )
            except:
                # Fallback to transformers ONNX export
                onnx_path = hf_hub_download(
                    repo_id="intfloat/multilingual-e5-small",
                    filename="onnx/model.onnx"
                )
        
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f"ONNX Embedder loaded from {onnx_path} with providers: {providers}")
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)
    
    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np"
            )
            
            # Run inference (some ONNX exports also require token_type_ids)
            feed = {
                'input_ids': encoded['input_ids'].astype(np.int64),
                'attention_mask': encoded['attention_mask'].astype(np.int64)
            }
            input_names = {inp.name for inp in self.session.get_inputs()}
            if 'token_type_ids' in input_names:
                feed['token_type_ids'] = np.zeros_like(encoded['input_ids'], dtype=np.int64)
            
            outputs = self.session.run(None, feed)
            
            token_embeddings = outputs[0]
            
            # Mean pooling
            input_mask_expanded = np.expand_dims(encoded['attention_mask'], -1).repeat(token_embeddings.shape[-1], axis=-1)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), 1e-9, None)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            embeddings = self._normalize(embeddings)
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text (optimized for query)."""
        return self.embed([text])[0]


# Global instance
_embedder = None

def get_embedder() -> ONNXEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ONNXEmbedder()
    return _embedder