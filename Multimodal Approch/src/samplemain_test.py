print("🔥 main.py loaded")

import os
from pipeline import run_pipeline

def main():
    print("✅ inside main()")

    query = "data/query/metal/metal.00004.wav"
    candidates_dir = "data/candidates/metal"   # same folder for now

    print("\nQuery song:", query)
    print("Scanning candidates in:", candidates_dir)

    for file in sorted(os.listdir(candidates_dir)):
        if not file.endswith(".wav"):
            continue

        candidate_path = os.path.join(candidates_dir, file)

        print("\n----------------------------------")
        print("Candidate:", file)

        report = run_pipeline(
            query_path=query,
            candidate_path=candidate_path,
            query_id="metal_00002",
            candidate_id=file.replace(".wav", "")
        )

        if report is None:
            print("Result: ❌ NO CONTAINMENT")
        else:
            print("Result: ✅ CONTAINMENT DETECTED")
            print("  Duration:", report["duration"])
            print("  Avg similarity:", round(report["average_similarity"], 3))
            print("  Segments matched:", report["segments_matched"])

    print("\n=== SCAN COMPLETE ===")

if __name__ == "__main__":
    main()
