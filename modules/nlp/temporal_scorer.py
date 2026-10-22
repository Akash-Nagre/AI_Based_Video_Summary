import numpy as np


def calculate_temporal_scores(segments: list) -> np.ndarray:
    """
    Calculate a temporal-context score for each segment.

    Segments near the beginning, middle, and end of the video
    receive slightly higher scores because these regions often
    contain important contextual information.
    """

    if not segments:
        return np.array([])

    video_start = segments[0]["start"]
    video_end = segments[-1]["end"]

    video_duration = video_end - video_start

    if video_duration <= 0:
        return np.ones(len(segments))

    scores = []

    for segment in segments:

        midpoint = (
            segment["start"] + segment["end"]
        ) / 2

        normalized_position = (
            midpoint - video_start
        ) / video_duration

        # Distance from the nearest major contextual region:
        # beginning, middle, or end.
        distances = [
            abs(normalized_position - 0.0),
            abs(normalized_position - 0.5),
            abs(normalized_position - 1.0)
        ]

        nearest_distance = min(distances)

        score = 1.0 - (
            nearest_distance / 0.5
        )

        scores.append(score)

    return np.array(scores)