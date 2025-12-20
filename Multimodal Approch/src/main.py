import os
import json
import sys

sys.path.append("src")
from pipeline import run_pipeline


QUERY_SONG = "data/query/metal/metal.00007.wav"
CANDIDATE_DIR = "data/query/metal"

FEATURES_DIR = "features"
GRAPHS_DIR = "graphs"
RESULTS_DIR = "results"


def ensure_dirs():
    os.makedirs(FEATURES_DIR + "/melody", exist_ok=True)
    os.makedirs(FEATURES_DIR + "/rhythm", exist_ok=True)
    os.makedirs(FEATURES_DIR + "/harmony", exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    print("\n🔥 FINAL SIMILARITY RUN (FIRST 10 SONGS) 🔥\n")

    ensure_dirs()

    candidates = [
        f for f in sorted(os.listdir(CANDIDATE_DIR))
        if f.endswith(".wav")
    ][:10]   # 🔴 LIMIT TO FIRST 10

    for idx, candidate_file in enumerate(candidates):
        print(f"\n[{idx+1}/10] Comparing with {candidate_file}")

        out = run_pipeline(
            query_path=QUERY_SONG,
            candidate_path=os.path.join(CANDIDATE_DIR, candidate_file),
            query_id="metal_00007",
            candidate_id=candidate_file.replace(".wav", "")
        )

        report = out["report"]
        q_segments = out["query_segments"]
        c_segments = out["candidate_segments"]
        matches = out["matches"]

        # =====================
        # SAVE FEATURES
        # =====================
        save_json(
            f"{FEATURES_DIR}/melody/query_metal_00007.json",
            [{ "id": s["segment_id"], "melody": s["melody"] } for s in q_segments]
        )

        save_json(
            f"{FEATURES_DIR}/rhythm/query_metal_00007.json",
            [{ "id": s["segment_id"], "rhythm": s["rhythm"] } for s in q_segments]
        )

        save_json(
            f"{FEATURES_DIR}/harmony/query_metal_00007.json",
            [{ "id": s["segment_id"], "harmony": s["harmony"] } for s in q_segments]
        )

        # =====================
        # SAVE GRAPH DATA
        # =====================
        save_json(
            f"{GRAPHS_DIR}/graph_metal_00007__{candidate_file}.json",
            {
                "matches": matches,
                "num_matches": len(matches)
            }
        )

        # =====================
        # SAVE FINAL RESULT
        # =====================
        if report:
            print("   ✅ CONTAINMENT DETECTED")
            save_json(
                f"{RESULTS_DIR}/metal_00007__{candidate_file}.json",
                report
            )
        else:
            print("   ❌ NO CONTAINMENT")
            save_json(
                f"{RESULTS_DIR}/metal_00007__{candidate_file}.json",
                {
                    "containment": False
                }
            )

    print("\n✅ DONE — FEATURES, GRAPHS & RESULTS SAVED\n")


if __name__ == "__main__":
    main()
