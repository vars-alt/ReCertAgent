import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticInspector:
    def __init__(self, name):
        self.model = SentenceTransformer(name)

    def max_similarity(self, texts, risk_descriptions):
        if not texts:
            return 0.0
        a = self.model.encode(texts, normalize_embeddings=True)
        b = self.model.encode(risk_descriptions, normalize_embeddings=True)
        return float(np.max(a @ b.T))
