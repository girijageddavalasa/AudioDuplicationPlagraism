import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from fastdtw import fastdtw

def dtw_sim(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d, _ = fastdtw(a, b)
    return np.exp(-d)

def fused_similarity(q, c):
    sm = dtw_sim(q["melody"], c["melody"])
    sr = dtw_sim(q["rhythm"], c["rhythm"])
    sh = 1 - cosine(q["harmony"], c["harmony"])
    return 0.5 * sm + 0.3 * sr + 0.2 * sh

def run_graph_matching(q_segments, c_segments, threshold=0.3):
    n, m = len(q_segments), len(c_segments)
    sim = np.zeros((n, m))

    for i, q in enumerate(q_segments):
        for j, c in enumerate(c_segments):
            s = fused_similarity(q, c)
            if s >= threshold:
                sim[i, j] = s

    cost = 1 - sim
    r, c = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(r, c):
        if sim[i, j] > 0:
            matches.append({
                "query": q_segments[i],
                "candidate": c_segments[j],
                "score": float(sim[i, j])
            })

    return matches
