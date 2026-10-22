import numpy as np


def calculate_multimodal_scores(
    text_scores: np.ndarray,
    visual_scores: np.ndarray,
    text_weight: float = 0.5,
    visual_weight: float = 0.5
) -> np.ndarray:
    """
    Combine text and visual importance scores.

    Both scores are expected to be normalized between 0 and 1.
    """

    if len(text_scores) != len(visual_scores):
        raise ValueError(
            "Text scores and visual scores must have "
            "the same number of segments."
        )

    final_scores = (
        text_weight * text_scores
        + visual_weight * visual_scores
    )

    return final_scores