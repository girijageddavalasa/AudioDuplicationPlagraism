def generate_final_report(query_song_id, candidate_song_id, match_records, query_song_duration):
    if not match_records:
        return None

    match_records.sort(key=lambda x: x["query"]["start"])

    q_start = match_records[0]["query"]["start"]
    q_end = match_records[-1]["query"]["end"]
    c_start = match_records[0]["candidate"]["start"]
    c_end = match_records[-1]["candidate"]["end"]

    duration = q_end - q_start
    avg_sim = sum(m["score"] for m in match_records) / len(match_records)

    if duration < 1 or avg_sim < 0.1:
        return None

    return {
        "query_song_id": query_song_id,
        "candidate_song_id": candidate_song_id,
        "query_time": [q_start, q_end],
        "candidate_time": [c_start, c_end],
        "duration": duration,
        "average_similarity": avg_sim,
        "segments_matched": len(match_records)
    }
