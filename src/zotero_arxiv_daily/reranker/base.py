from abc import ABC, abstractmethod
from datetime import timezone
from typing import Type

import numpy as np
from omegaconf import DictConfig

from ..protocol import Paper, CorpusPaper


class BaseReranker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        if not candidates:
            return []
        if not corpus:
            raise ValueError("A non-empty Zotero corpus is required for ranking")
        corpus = sorted(corpus, key=lambda p: self._utc_date(p), reverse=True)
        options = self.config.reranker.get("ranking", {}) if self.config is not None else {}
        mode = options.get("mode", "legacy")
        if mode not in {"legacy", "theme_balanced"}:
            raise ValueError(f"Unsupported ranking mode: {mode}")
        texts = [p.abstract for p in candidates]
        corpus_texts = [p.abstract for p in corpus]
        # A single embedding call also provides candidate-to-candidate similarity.
        sim = self.get_similarity_score(texts, corpus_texts + (texts if mode == "theme_balanced" else []))
        expected = (len(candidates), len(corpus) + (len(candidates) if mode == "theme_balanced" else 0))
        if sim.shape != expected or not np.isfinite(sim).all():
            raise ValueError("Invalid embedding similarity matrix")
        # Kept so a run's ranking can be archived and replayed offline without the model.
        self.last_run = {"candidates": candidates, "corpus": corpus, "similarity": sim}
        if mode == "theme_balanced":
            return self._theme_rerank(candidates, corpus, sim, options)
        weights = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        weights /= weights.sum()
        scores = (sim * weights).sum(axis=1) * 10
        for i, paper in enumerate(candidates):
            paper.score = float(scores[i])
            paper.selection_score = paper.score
            paper.recommendation_theme = None
            paper.related_papers = [corpus[j].title for j in np.argsort(-sim[i], kind="stable")[:3]]
        return sorted(candidates, key=lambda p: p.score, reverse=True)

    @staticmethod
    def _utc_date(paper):
        date = paper.added_date
        return date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc)

    def _theme_rerank(self, candidates, corpus, similarity, options):
        half_life = float(options.get("recency_half_life_days", 180))
        depth = int(options.get("theme_path_depth", 2))
        top_k = int(options.get("neighbors_per_theme", 3))
        theme_penalty = float(options.get("theme_penalty", 0.04))
        diversity_weight = float(options.get("diversity_weight", 0.15))
        duplicate_threshold = float(options.get("duplicate_similarity_threshold", 0.85))
        if (half_life <= 0 or depth < 1 or top_k < 1 or not 0 <= theme_penalty <= 1
                or not 0 <= diversity_weight <= 1 or not 0 <= duplicate_threshold < 1):
            raise ValueError("Invalid theme ranking parameters")
        groups = {}
        for j, paper in enumerate(corpus):
            # Collections outside include_path must not form themes or win recommendations.
            paths = paper.paths if paper.theme_paths is None else paper.theme_paths
            themes = {"/".join(path.split("/")[:depth]) for path in paths if path}
            for theme in sorted(themes or {"未分类"}):
                groups.setdefault(theme, []).append(j)
        names = sorted(groups)
        corpus_sim = similarity[:, :len(corpus)]
        newest = self._utc_date(corpus[0])
        ages = np.array([(newest - self._utc_date(p)).total_seconds() / 86400 for p in corpus])
        theme_scores = np.empty((len(candidates), len(names)))
        neighbors = {}
        for t, theme in enumerate(names):
            indices = np.array(groups[theme])
            for i in range(len(candidates)):
                nearest = indices[np.argsort(-corpus_sim[i, indices], kind="stable")[:top_k]]
                # Relative ages avoid underflow and make the score independent of run date.
                weights = np.exp2(-(ages[nearest] - ages[nearest].min()) / half_life)
                theme_scores[i, t] = np.average(corpus_sim[i, nearest], weights=weights) * 10
                neighbors[i, t] = nearest
        best_theme = theme_scores.argmax(axis=1)
        scores = theme_scores[np.arange(len(candidates)), best_theme]
        candidate_sim = similarity[:, len(corpus):]
        selected, remaining = [], list(range(len(candidates)))
        counts = np.zeros(len(names), dtype=int)
        redundancy = np.zeros(len(candidates))
        while remaining:
            utility = scores - 10 * theme_penalty * counts[best_theme] - 10 * diversity_weight * redundancy
            i = max(remaining, key=lambda j: (utility[j], scores[j], -j))
            paper = candidates[i]
            t = best_theme[i]
            paper.score = float(scores[i])
            paper.selection_score = float(utility[i])
            paper.recommendation_theme = names[t]
            paper.related_papers = [corpus[j].title for j in neighbors[i, t]]
            selected.append(paper)
            remaining.remove(i)
            counts[t] += 1
            duplicate = np.clip((candidate_sim[:, i] - duplicate_threshold) / (1 - duplicate_threshold), 0, 1)
            redundancy = np.maximum(redundancy, duplicate)
        return selected

    @abstractmethod
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError


registered_rerankers = {}


def register_reranker(name: str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator


def get_reranker_cls(name: str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
