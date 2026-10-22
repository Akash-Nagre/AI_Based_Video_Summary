import numpy as np


def calculate_redundancy_scores(
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Calculate semantic redundancy for each candidate segment.

    A segment receives a higher redundancy score when it is
    semantically similar to another candidate segment.

    Embeddings are normalized, so their dot product gives
    cosine similarity.
    """

    number_of_segments = len(embeddings)

    if number_of_segments == 0:
        return np.array([])

    if number_of_segments == 1:
        return np.zeros(1)

    similarity_matrix = np.dot(
        embeddings,
        embeddings.T
    )

    redundancy_scores = np.zeros(number_of_segments)

    for i in range(number_of_segments):

        similarities = similarity_matrix[i].copy()

        # Ignore similarity with itself.
        similarities[i] = -1.0

        max_similarity = np.max(similarities)

        # Convert similarity from [-1, 1] to [0, 1].
        redundancy_scores[i] = (
            max_similarity + 1
        ) / 2

    return redundancy_scores