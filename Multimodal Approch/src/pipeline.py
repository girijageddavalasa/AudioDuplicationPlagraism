from audio_loader import load_audio
from segmentation import segment_audio
from feature_extraction import extract_features
from graph_matching import run_graph_matching
from containment import generate_final_report

def run_pipeline(query_path, candidate_path, query_id, candidate_id):
    q_audio, sr, q_dur = load_audio(query_path)
    c_audio, _, _ = load_audio(candidate_path)

    q_segments_raw = segment_audio(q_audio, sr)
    c_segments_raw = segment_audio(c_audio, sr)

    q_segments = extract_features(q_segments_raw, sr, "Q")
    c_segments = extract_features(c_segments_raw, sr, "C")

    matches = run_graph_matching(q_segments, c_segments)

    report = generate_final_report(
        query_id, candidate_id, matches, q_dur
    )

    return {
        "report": report,
        "query_segments": q_segments,
        "candidate_segments": c_segments,
        "matches": matches
    }
