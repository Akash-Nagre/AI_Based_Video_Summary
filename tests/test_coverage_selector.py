import numpy as np

from modules.multimodal.highlight_selector import select_highlights


def test_duration_budget_and_topic_marginal_gain():
    segments = [
        {"start": 0, "end": 5, "text": "Topic A explanation one"},
        {"start": 20, "end": 25, "text": "Topic A repeated explanation"},
        {"start": 40, "end": 45, "text": "Topic B important explanation"},
    ]
    embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=float)
    selected = select_highlights(segments, np.asarray([0.95, 0.90, 0.40]), embeddings,
        number_of_highlights=3, max_duration_seconds=10, topic_ids=[0, 0, 1],
        topic_importance={0: 0.6, 1: 0.4}, minimum_topic_seconds={0: 5, 1: 5})
    assert sum(x["end"] - x["start"] for x in selected) <= 10
    assert [x["topic_id"] for x in selected] == [0, 1]


def test_selected_clips_are_chronological_and_non_overlapping():
    segments = [{"start": 0, "end": 6}, {"start": 5, "end": 10}, {"start": 20, "end": 25}]
    selected = select_highlights(segments, np.asarray([1.0, 0.9, 0.8]), np.eye(3), number_of_highlights=3)
    assert [item["start"] for item in selected] == [0, 20]
