import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Union
import logging
from transformers import AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)


class ONNXEmbedder:
    """Fast embedder for multilingual-e5-small (uses transformers directly for now)."""
    
    def __init__(self, model_path: str = None, providers: List[str] = None):
        # Use local model if available
        if model_path is None:
            local_path = Path("backend/models/multilingual-e5-small")
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = "intfloat/multilingual-e5-small"
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2 normalize embeddings."""
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        return embeddings / torch.clamp(norms, min=1e-12)
    
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
                return_tensors="pt"
            )
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                
                # Mean pooling
                embeddings = self._mean_pooling(token_embeddings, encoded["attention_mask"])
                
                # Normalize
                embeddings = self._normalize(embeddings)
            
            all_embeddings.append(embeddings.numpy())
        
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