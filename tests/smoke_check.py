import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from app.sim import equity_vs_random_opponents
import eval7

def run_smoke_test():
    hero = [eval7.Card("As"), eval7.Card("Ad")]

    result = equity_vs_random_opponents(hero, [], 1, 2000)
    hero_equity = result[0] if isinstance(result, (list, tuple)) else result["equity"]

    print("AA Equity:", hero_equity)

    if 0.60 < hero_equity < 0.95:
        print("✅ Smoke Test Passed")
    else:
        print("❌ Smoke Test Failed")

if __name__ == "__main__":
    run_smoke_test()
