"""Transcript-derived, domain-agnostic topic grouping and coverage metrics."""
from collections import Counter
import math
import re
import numpy as np

_COMMON = set("a an the and or but if then this that these those it its to of in on at for by with from as is are was were be been being i we you they he she them our your their have has had do did does not no yes so very just about into over after before than what when where why how who which there here can could should would will may might also".split())


def _label(texts, all_topic_texts):
    """Choose a source phrase with corpus-level IDF; never synthesize facts."""
    docs = [set(re.findall(r"[\w'-]{3,}", x.lower())) - _COMMON for x in all_topic_texts]
    counts = Counter()
    for text in texts:
        tokens = re.findall(r"[\w'-]{3,}", text.lower())
        for token in tokens:
            if token not in _COMMON:
                counts[token] += 1
    if not counts:
        return texts[0][:90]
    score = lambda token: counts[token] * math.log(1 + len(docs) / (1 + sum(token in d for d in docs)))
    top = [t for t, _ in sorted(counts.items(), key=lambda item: (-score(item[0]), item[0]))[:3]]
    # Find the shortest source sentence containing the top terms; otherwise quote representative text.
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(texts)):
        low = sentence.lower()
        if sum(t in low for t in top[:2]) >= min(2, len(top)):
            return sentence[:100]
    return texts[0][:100]


def discover_topics(segments, embeddings, importance_scores):
    """Greedy cosine clustering, with no fixed topic count or domain vocabulary."""
    vectors = np.asarray(embeddings, dtype=float)
    if len(segments) != len(vectors):
        raise ValueError("Segments and embeddings must have equal lengths")
    if not segments:
        return [], []
    # Similarity threshold adapts modestly to how cohesive nearby transcript units are.
    sims = [float(vectors[i] @ vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = float(np.clip(np.median(sims) - 0.04, 0.48, 0.68)) if sims else 0.55
    centers, counts, assignments = [], [], []
    for vector in vectors:
        if not centers:
            centers.append(vector.copy()); counts.append(1); assignments.append(0); continue
        similarities = np.asarray([float(vector @ center) for center in centers])
        topic_id = int(np.argmax(similarities))
        if similarities[topic_id] < threshold:
            centers.append(vector.copy()); counts.append(1); assignments.append(len(centers) - 1)
        else:
            counts[topic_id] += 1
            centers[topic_id] = centers[topic_id] + (vector - centers[topic_id]) / counts[topic_id]
            norm = np.linalg.norm(centers[topic_id])
            if norm: centers[topic_id] /= norm
            assignments.append(topic_id)
    all_texts = [" ".join(s["text"] for s, aid in zip(segments, assignments) if aid == t) for t in range(len(centers))]
    raw = []
    for tid, center in enumerate(centers):
        indices = [i for i, assigned in enumerate(assignments) if assigned == tid]
        members = [segments[i] for i in indices]
        duration = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in members)
        salience = float(np.mean([importance_scores[i] for i in indices]))
        spread = len({int(3 * ((float(s["start"]) - float(segments[0]["start"])) / max(1.0, float(segments[-1]["end"]) - float(segments[0]["start"])))) for s in members})
        coherence = float(np.mean([vectors[i] @ center for i in indices]))
        raw.append({"topic_id": tid, "label": _label([s["text"] for s in members], all_texts),
            "description": " ".join(s["text"] for s in members[:3]), "start": float(members[0]["start"]),
            "end": float(members[-1]["end"]), "segment_indices": indices, "segment_count": len(members),
            "duration_seconds": duration, "salience": salience, "temporal_spread": spread,
            "coherence": coherence, "importance": max(0.01, 0.65 * salience + 0.20 * min(1.0, spread / 3) + 0.15 * coherence),
            "confidence": coherence, "subtopics": []})
    importance_sum = sum(t["importance"] for t in raw) or 1.0
    for topic in raw:
        topic["importance_share"] = topic["importance"] / importance_sum
        topic["evidence"] = [segments[i]["text"] for i in topic["segment_indices"]]
        topic["source_timestamps"] = [[float(segments[i]["start"]), float(segments[i]["end"])] for i in topic["segment_indices"]]
    return raw, assignments


def assess_topic_coverage(topics, selected_segments, minimum_topic_seconds):
    """Coverage is capped selected duration / estimated useful duration, weighted by salience.

    This is an evidence-duration proxy; it does not establish factual completeness.
    """
    result = []
    for topic in topics:
        chosen = [s for s in selected_segments if s.get("topic_id") == topic["topic_id"]]
        selected_seconds = sum(float(s["end"]) - float(s["start"]) for s in chosen)
        need = max(1.0, float(minimum_topic_seconds[topic["topic_id"]]))
        coverage = min(1.0, selected_seconds / need)
        status = "adequately_represented" if coverage >= 0.75 else "partially_covered" if coverage > 0 else "omitted_due_to_duration"
        result.append({**topic, "selected_clips": [[s["start"], s["end"]] for s in chosen],
            "selected_evidence": [s["text"] for s in chosen], "selected_duration_seconds": selected_seconds,
            "minimum_useful_seconds": need, "coverage_score": coverage, "coverage_status": status,
            "omission_reason": "duration_budget_or_low_marginal_value" if not chosen else None})
    return result
