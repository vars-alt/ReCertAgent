import torch
from transformers import pipeline

class PIGuard:
    def __init__(self, name="leolee99/PIGuard"):
        self.pipe = pipeline(
            "text-classification",
            model=name,
            tokenizer=name,
            trust_remote_code=True,
            truncation=True,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,
        )

    def injection_score(self, texts):
        if not texts:
            return 0.0
        outputs = self.pipe(texts)
        scores = []
        for output in outputs:
            rows = output if isinstance(output, list) else [output]
            malicious = 0.0
            for row in rows:
                label = str(row["label"]).lower()
                if label in {"label_1", "1", "injection", "malicious"}:
                    malicious = max(malicious, float(row["score"]))
            scores.append(malicious)
        return max(scores, default=0.0)
