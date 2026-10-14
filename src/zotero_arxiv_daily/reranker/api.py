from .base import BaseReranker, register_reranker
from openai import OpenAI
import numpy as np
@register_reranker("api")
class ApiReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        client = OpenAI(api_key=self.config.reranker.api.key, base_url=self.config.reranker.api.base_url)
        configured_batch_size = self.config.reranker.api.get('batch_size')
        batch_size = 64 if configured_batch_size is None else int(configured_batch_size)
        if batch_size <= 0:
            raise ValueError('Embedding batch_size must be positive')
        all_texts = list(dict.fromkeys(s1 + s2))
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.config.reranker.api.model
            )
            data = list(response.data)
            if len(data) != len(batch):
                raise ValueError('Embedding response omitted inputs')
            if any(hasattr(item, 'index') for item in data):
                if (not all(hasattr(item, 'index') for item in data)
                        or sorted(item.index for item in data) != list(range(len(batch)))):
                    raise ValueError('Embedding response has invalid indices')
                data.sort(key=lambda item: item.index)
            all_embeddings.extend([item.embedding for item in data])
        indices = {text: i for i, text in enumerate(all_texts)}
        s1_embeddings = np.array([all_embeddings[indices[text]] for text in s1])
        s2_embeddings = np.array([all_embeddings[indices[text]] for text in s2])
        norms1 = np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        norms2 = np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        if (not np.isfinite(s1_embeddings).all() or not np.isfinite(s2_embeddings).all()
                or not np.isfinite(norms1).all() or not np.isfinite(norms2).all()
                or (norms1 == 0).any() or (norms2 == 0).any()):
            raise ValueError('Embedding response contains nonfinite or zero vectors')
        s1_embeddings_normalized = s1_embeddings / norms1
        s2_embeddings_normalized = s2_embeddings / norms2
        sim = np.dot(s1_embeddings_normalized, s2_embeddings_normalized.T) # [n_s1, n_s2]
        return sim
