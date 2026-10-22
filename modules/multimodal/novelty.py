import numpy as np


def calculate_novelty_scores(
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate novelty scores for candidate segments.

    A segment receives a lower novelty score when it is
    highly similar to another candidate segment.

    Higher score = more unique information.
    """

    if len(embeddings) == 0:
        return np.array([])

    if len(embeddings) == 1:
        return np.ones(1)

    # Embeddings are already normalized.
    similarity_matrix = np.dot(
        embeddings,
        embeddings.T
    )

    # Ignore similarity of a segment with itself.
    np.fill_diagonal(
        similarity_matrix,
        -1.0
    )

    # Find the strongest similarity to another segment.
    max_similarity = np.max(
        similarity_matrix,
        axis=1
    )

    # Convert similarity into novelty.
    #
    # High similarity → low novelty
    # Low similarity  → high novelty
    novelty_scores = 1.0 - max_similarity

    # Normalize to 0–1.
    min_score = novelty_scores.min()
    max_score = novelty_scores.max()

    if max_score != min_score:
        novelty_scores = (
            (novelty_scores - min_score)
            / (max_score - min_score)
        )
    else:
        novelty_scores = np.ones(
            len(novelty_scores)
        )

    return novelty_scores