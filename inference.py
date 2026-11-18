"""Inference using the eight features and Viterbi settings in the project notebook."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURE_COLUMNS = [
    "distance_to_segment", "segment_length", "projection_position",
    "gps_speed", "time_delta", "gps_bearing", "segment_bearing",
    "heading_difference",
]
TOP_K = 12
SEARCH_K = 30
EARTH_RADIUS = 6371000.0
JUMP_PENALTY = 4.0
CONNECTED_PENALTY = 0.5
GAP_RESET_SECONDS = 10.0


def read_trip(folder: Path):
    trip_id = folder.name
    track = np.loadtxt(folder / f"{trip_id}.track", ndmin=2)
    nodes = np.loadtxt(folder / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(folder / f"{trip_id}.arcs", dtype=int, ndmin=2)
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Trip file has invalid columns")
    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise ValueError("Empty trip files")
    if not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("Nonfinite GPS or road coordinates")
    return track, nodes, arcs


def to_metres(coords, reference_lat):
    lon = np.radians(coords[:, 0])
    lat = np.radians(coords[:, 1])
    return np.column_stack((EARTH_RADIUS * lon * np.cos(reference_lat), EARTH_RADIUS * lat))


def build_candidates(track, nodes, arcs):
    """Mirror notebook cell 38, preserving original arc IDs and feature order."""
    valid = ((arcs[:, 0] >= 0) & (arcs[:, 1] >= 0) &
             (arcs[:, 0] < len(nodes)) & (arcs[:, 1] < len(nodes)))
    original_ids = np.flatnonzero(valid)
    if not len(original_ids):
        raise ValueError("Road network has no valid arcs")
    road_arcs = arcs[valid]
    reference_lat = np.radians(track[:, 1].mean())
    gps = to_metres(track[:, :2], reference_lat)
    starts = to_metres(nodes[road_arcs[:, 0], :2], reference_lat)
    ends = to_metres(nodes[road_arcs[:, 1], :2], reference_lat)
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    bearings = np.arctan2(vectors[:, 0], vectors[:, 1])
    search = min(SEARCH_K, len(road_arcs))
    width = min(TOP_K, search)
    _, nearby = cKDTree((starts + ends) / 2).query(gps, k=search)
    nearby = np.asarray(nearby).reshape(len(gps), search)
    a, v = starts[nearby], vectors[nearby]
    denominator = np.maximum(np.sum(v * v, axis=2), 1e-12)
    projection_all = np.clip(
        np.sum((gps[:, None, :] - a) * v, axis=2) / denominator, 0, 1
    )
    closest = a + projection_all[:, :, None] * v
    distances_all = np.linalg.norm(gps[:, None, :] - closest, axis=2)
    chosen = np.argsort(distances_all, axis=1)[:, :width]
    positions = np.take_along_axis(nearby, chosen, axis=1)
    distances = np.take_along_axis(distances_all, chosen, axis=1)
    projection = np.take_along_axis(projection_all, chosen, axis=1)

    delta = np.zeros_like(gps)
    delta[1:] = np.diff(gps, axis=0)
    dt = np.zeros(len(gps))
    dt[1:] = np.diff(track[:, 2])
    speed = np.zeros(len(gps))
    positive = dt > 0
    speed[positive] = np.linalg.norm(delta[positive], axis=1) / dt[positive]
    gps_bearing = np.arctan2(delta[:, 0], delta[:, 1])
    repeated_bearing = np.repeat(gps_bearing[:, None], width, axis=1)
    segment_bearing = bearings[positions]
    heading = np.abs((repeated_bearing - segment_bearing + np.pi) % (2 * np.pi) - np.pi)

    frame = pd.DataFrame({
        "distance_to_segment": distances.ravel(),
        "segment_length": lengths[positions].ravel(),
        "projection_position": projection.ravel(),
        "gps_speed": np.repeat(speed, width),
        "time_delta": np.repeat(dt, width),
        "gps_bearing": repeated_bearing.ravel(),
        "segment_bearing": segment_bearing.ravel(),
        "heading_difference": heading.ravel(),
    })[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return original_ids[positions], frame


def viterbi(ids, probabilities, arcs, timestamps):
    """Mirror the tuned RF + HMM notebook cell, including arc-ID sort."""
    ordered = np.argsort(ids, axis=1)
    candidates = np.take_along_axis(ids, ordered, axis=1)
    probabilities = np.take_along_axis(probabilities, ordered, axis=1)
    n, k = candidates.shape
    emission = np.log(np.clip(probabilities, 1e-12, 1.0))
    scores = emission[0].copy()
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        previous, current = candidates[t - 1], candidates[t]
        same = previous[:, None] == current[None, :]
        connected = arcs[previous, 1, None] == arcs[current, 0][None, :]
        transition = np.where(same, 0.0, np.where(connected, -CONNECTED_PENALTY, -JUMP_PENALTY))
        dt = timestamps[t] - timestamps[t - 1]
        if dt <= 0 or dt > GAP_RESET_SECONDS:
            transition[:] = 0.0
        choices = scores[:, None] + transition
        back[t] = np.argmax(choices, axis=0)
        scores = emission[t] + np.max(choices, axis=0)
    states = np.zeros(n, dtype=int)
    states[-1] = np.argmax(scores)
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return candidates[np.arange(n), states]


def route_continuity(predicted, arcs):
    if len(predicted) < 2:
        return None
    joins = ((predicted[:-1] == predicted[1:]) |
             (arcs[predicted[:-1], 1] == arcs[predicted[1:], 0]))
    return float(joins.mean())


def predict(folder: Path, model):
    track, nodes, arcs = read_trip(folder)
    ids, features = build_candidates(track, nodes, arcs)
    probability = model.predict_proba(features)[:, 1].reshape(ids.shape)
    classifier = ids[np.arange(len(ids)), probability.argmax(axis=1)]
    hmm = viterbi(ids, probability, arcs, track[:, 2])
    metrics = {"observations": len(track), "route_continuity": route_continuity(hmm, arcs)}

    # Evaluation only. The reference route is never supplied to the model or UI.
    route_path = folder / f"{folder.name}.route"
    if route_path.is_file():
        from reference_evaluation import reference_labels
        target, confident = reference_labels(track, nodes, arcs, route_path)
        metrics["evaluated_observations"] = int(confident.sum())
        metrics["classifier_accuracy"] = float(np.mean(classifier[confident] == target[confident])) if confident.any() else None
        metrics["hmm_accuracy"] = float(np.mean(hmm[confident] == target[confident])) if confident.any() else None

    # Each road segment is a [longitude, latitude] pair of endpoints.
    return {"trip_id": folder.name, "metrics": metrics,
            "gps_points": track[:, :2].tolist(),
            "predicted_arc_ids": hmm.tolist(),
            "predicted_segments": nodes[arcs[hmm], :2].tolist()}
