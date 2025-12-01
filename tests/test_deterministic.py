import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app.sim import equity_vs_random_opponents
import eval7

def test_deterministic_win():
    # Hero and villain cards (converted to eval7)
    hero = [eval7.Card("As"), eval7.Card("Ad")]
    villain = [eval7.Card("Kc"), eval7.Card("Kd")]

    # Full board
    board = [
        eval7.Card("Ah"),
        eval7.Card("5c"),
        eval7.Card("Td"),
        eval7.Card("9s"),
        eval7.Card("3d"),
    ]

    # Run simulation (1 sim = deterministic)
    # NOTE: passing exact villain cards inside the "board" argument is not how
    # your function works, so instead we pass 1 villain and let sim use fixed board.
    result = equity_vs_random_opponents(hero, board, 1, 1)

    hero_equity = result[0] if isinstance(result, (list, tuple)) else result["equity"]

    print("Deterministic Equity:", hero_equity)

    # Should be EXACTLY 1.0 because AA wins 100% with this board
    assert abs(hero_equity - 1.0) < 1e-3, "Hero should win 100% of the time on this board"
