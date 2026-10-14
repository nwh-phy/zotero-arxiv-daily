from .base import BaseReranker, register_reranker
import logging
import warnings
import numpy as np
@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True,
                                      device=self.config.reranker.local.get("device", "cpu"))
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        texts = list(dict.fromkeys(s1 + s2))
        features = encoder.encode(texts, **encode_kwargs, show_progress_bar=True)
        indices = {text: i for i, text in enumerate(texts)}
        s1_feature = features[[indices[text] for text in s1]]
        s2_feature = features[[indices[text] for text in s2]]
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
