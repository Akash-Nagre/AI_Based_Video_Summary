
import re
import hashlib
import warnings
from functools import lru_cache

import numpy as np


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load semantic embeddings; use deterministic lexical fallback offline."""
    try:
        from sentence_transformers import SentenceTransformer
        try:
            # Most repeat jobs already have model weights on disk; avoid Hub
            # connectivity retries before loading the local model.
            return SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            # Preserve first-run model download behavior when no local copy exists.
            return SentenceTransformer(MODEL_NAME)
    except Exception as exc:
        warnings.warn(f"Sentence embedding model unavailable; using lexical hash vectors: {exc}")
        class LexicalHashModel:
            embedding_fallback = True
            dimension = 384
            def get_sentence_embedding_dimension(self): return self.dimension
            def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False):
                output = np.zeros((len(texts), self.dimension), dtype=np.float32)
                for row, text in enumerate(texts):
                    tokens = re.findall(r"[\w'-]+", str(text).lower())
                    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
                    for feature in features:
                        digest = hashlib.sha256(feature.encode("utf-8")).digest()
                        index = int.from_bytes(digest[:4], "little") % self.dimension
                        output[row, index] += 1.0 if digest[4] & 1 else -1.0
                    norm = np.linalg.norm(output[row])
                    if norm: output[row] /= norm
                return output
        return LexicalHashModel()


def generate_embeddings(segments: list, model) -> np.ndarray:
    """Generate normalized semantic embeddings."""

    if not segments:
        dimension = model.get_sentence_embedding_dimension() or 384
        return np.empty((0, dimension), dtype=np.float32)

    texts = [
        str(segment.get("text", "")).strip()
        for segment in segments
    ]

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )


def _normalize_scores(scores) -> np.ndarray:
    """Normalize scores to the 0–1 range."""

    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        return scores

    low = scores.min()
    high = scores.max()

    if high - low < 1e-10:
        return np.full(len(scores), 0.5)

    return (scores - low) / (high - low)


def _educational_cue_score(text: str) -> float:
    """
    Estimate educational importance from explanatory language.

    This is a heuristic, not full language understanding.
    """

    text = text.lower().strip()

    if not text:
        return 0.0

    patterns = [
        r"\b(defined as|definition of|refers to|means|"
        r"is known as|is called)\b",

        r"\b(principle|law of|theorem|formula|equation|"
        r"according to|archimedes|buoyant force)\b",

        r"\b(because|therefore|this means|as a result|"
        r"the reason|hence|consequently|due to)\b",

        r"\b(for example|for instance|consider|suppose|"
        r"let us take|imagine|in this case)\b",

        r"\b(in conclusion|to summarize|in summary|"
        r"key point|remember that|finally)\b",

        r"\b(important|significant|note that|"
        r"must remember|key concept|main idea)\b",
    ]

    # Ignore segments that contain only common filler phrases.
    if re.fullmatch(
        r"[\W]*(okay|ok|right|yeah|yes|so|umm|um|"
        r"uh|alright|thank you)[\W]*",
        text,
    ):
        return 0.0

    score = sum(
        1.0 for pattern in patterns
        if re.search(pattern, text)
    )

    word_count = len(text.split())

    if word_count >= 8:
        score += 0.5

    if word_count >= 15:
        score += 0.5

    return score


def calculate_text_scores(
    embeddings: np.ndarray,
    damping: float = 0.85,
    iterations: int = 30,
    segments: list | None = None,
    centrality_weight: float = 0.55,
    educational_weight: float = 0.45,
) -> np.ndarray:
    """
    Combine semantic centrality and educational-language cues.
    """

    embeddings = np.asarray(embeddings, dtype=float)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings must contain finite values.")

    if len(embeddings) == 0:
        return np.array([], dtype=float)

    if not np.isfinite(damping) or not 0 <= damping < 1:
        raise ValueError("damping must be in the range [0, 1).")

    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer.")

    if (
        not np.isfinite(centrality_weight)
        or not np.isfinite(educational_weight)
        or centrality_weight < 0
        or educational_weight < 0
        or not np.isclose(centrality_weight + educational_weight, 1.0)
    ):
        raise ValueError("Text-score weights must be non-negative and sum to 1.")

    if len(embeddings) == 1:
        centrality = np.array([0.5])
    else:
        similarity = embeddings @ embeddings.T
        similarity = np.maximum(similarity, 0.0)
        np.fill_diagonal(similarity, 0.0)

        row_sums = similarity.sum(axis=1, keepdims=True)

        transition = np.divide(
            similarity,
            row_sums,
            out=np.zeros_like(similarity, dtype=float),
            where=row_sums != 0,
        )

        count = len(embeddings)
        scores = np.full(count, 1.0 / count)

        for _ in range(iterations):
            scores = (
                (1.0 - damping) / count
                + damping * transition.T @ scores
            )

        centrality = _normalize_scores(scores)

    # Keep backward compatibility with the old function call.
    if segments is None:
        return centrality

    if len(segments) != len(embeddings):
        raise ValueError("Segments and embeddings must have equal lengths.")

    cue_scores = _normalize_scores([
        _educational_cue_score(str(segment.get("text", "")))
        for segment in segments
    ])

    return (
        centrality_weight * centrality
        + educational_weight * cue_scores
    )


def calculate_length_scores(
    segments: list,
    ideal_duration: float = 12.0,
) -> np.ndarray:
    """Score how suitable each segment's duration is."""

    if ideal_duration <= 0:
        raise ValueError("ideal_duration must be positive.")

    if not segments:
        return np.array([], dtype=float)

    scores = []

    for segment in segments:
        duration = float(segment["end"]) - float(segment["start"])

        if duration <= 0:
            score = 0.0
        else:
            score = float(
                np.exp(-abs(duration - ideal_duration) / ideal_duration)
            )

        scores.append(score)

    return np.asarray(scores, dtype=float)


def calculate_combined_text_scores(
    semantic_scores: np.ndarray,
    length_scores: np.ndarray,
    semantic_weight: float = 0.80,
    length_weight: float = 0.20,
) -> np.ndarray:
    """Combine semantic importance and duration suitability."""

    semantic_scores = np.asarray(semantic_scores, dtype=float)
    length_scores = np.asarray(length_scores, dtype=float)

    if semantic_scores.ndim != 1 or length_scores.ndim != 1:
        raise ValueError("Score arrays must be one-dimensional.")

    if len(semantic_scores) != len(length_scores):
        raise ValueError("Score arrays must have equal lengths.")

    if not np.isclose(semantic_weight + length_weight, 1.0):
        raise ValueError("Weights must sum to 1.")

    if semantic_weight < 0 or length_weight < 0:
        raise ValueError("Weights must be non-negative.")

    if not (
        np.all(np.isfinite(semantic_scores))
        and np.all(np.isfinite(length_scores))
    ):
        raise ValueError("Scores must be finite.")

    return (
        semantic_weight * semantic_scores
        + length_weight * length_scores
    )
