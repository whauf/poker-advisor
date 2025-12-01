import os
import sys
import json

# --- Ensure project root is importable ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app.ranges import load_ranges, in_range


def test_range_file_loads():
    # Correct JSON path
    json_path = os.path.join(ROOT, "data", "ranges", "preflop_live_fullring_100bb.json")

    assert os.path.exists(json_path), f"Ranges JSON not found at: {json_path}"

    ranges = load_ranges(json_path)

    # Basic sanity checks
    assert isinstance(ranges, dict), "Ranges should load into a dictionary"
    assert len(ranges) > 0, "Ranges should not be empty"

    # Expected positions exist
    for pos in ["BTN", "CO", "HJ", "UTG"]:
        assert pos in ranges, f"Missing position: {pos}"
        assert "open" in ranges[pos], f"Missing 'open' key for {pos}"

    # Spot-check a known combo
    assert in_range(ranges, "BTN", "AKs"), "AKs should be in BTN opening range"
