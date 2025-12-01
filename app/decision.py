# app/decision.py

import eval7
import math
import re
import random
from itertools import combinations
from .sim import equity_vs_random_opponents
from .ranges import load_ranges, in_range

# -------------------------------
# Preflop Hand Groups
# -------------------------------
PREMIUM_HANDS = {"AA", "KK", "QQ", "JJ", "AKs"}

STRONG_HANDS = {
    "AKo", "AQs", "AJs", "KQs", "TT", "99"
}

MEDIUM_HANDS = {
    "KQo", "AQo", "KJs", "QJs", "JTs",
    "88", "77", "ATs"
}

WEAK_HANDS = {
    "66", "55", "44", "A9s", "A8s",
    "KTs", "QTs", "J9s"
}

TRASH_HANDS = {
    "22", "33", "A2o", "K9o", "Q9o",
    "J8o", "T8o"
}

RANK_ORDER = "AKQJT98765432"
VALID_RANKS = set("23456789TJQKA")
VALID_SUITS = set("cdhs")


def _normalize_card_str(card: str) -> str:
    """
    Normalize a single card string into eval7 format: RankUpper + SuitLower.
    Accepts things like 'as', 'AS', 'Ah', 'AD', etc.
    """
    card = card.strip()
    if len(card) < 2:
        raise ValueError(f"Card '{card}' too short")

    rank = card[0].upper()
    suit = card[1].lower()

    if rank not in VALID_RANKS:
        raise ValueError(f"Invalid rank in card '{card}'")
    if suit not in VALID_SUITS:
        raise ValueError(f"Invalid suit in card '{card}' (must be one of c,d,h,s)")

    return rank + suit


def _normalize_cards_any(raw, expected_count: int | None = None) -> list[str]:
    """
    Accept:
      - list: ['As','Kd']
      - string: 'As Kd'
    Return normalized eval7 codes.
    """
    if raw is None:
        raise ValueError("No cards supplied")

    if isinstance(raw, str):
        tokens = raw.replace(",", " ").split()
    else:
        tokens = list(raw)

    norm = [_normalize_card_str(t) for t in tokens]

    if expected_count is not None and len(norm) != expected_count:
        raise ValueError(f"Expected {expected_count} cards, got {len(norm)}: {norm}")

    return norm


# -------------------------------
# Hand classification + draws
# -------------------------------
RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK_CODES = [r + s for r in RANKS for s in SUITS]

HAND_ORDER = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]
HAND_RANK_INDEX = {name: i for i, name in enumerate(HAND_ORDER)}


def _codes_to_cards_simple(codes):
    """
    ['As','Kd'] -> [(14,'s'), (13,'d')] etc
    """
    result = []
    rank_map = {r: i + 2 for i, r in enumerate(RANKS)}
    for c in codes:
        c = c.strip()
        if len(c) != 2:
            continue
        r, s = c[0].upper(), c[1].lower()
        if r not in rank_map or s not in SUITS:
            continue
        result.append((rank_map[r], s))
    return result


def _is_straight_5(ranks):
    rset = sorted(set(ranks))
    # Wheel
    if set(rset) == {14, 5, 4, 3, 2}:
        return True
    if len(rset) != 5:
        return False
    return max(rset) - min(rset) == 4


def _classify_5(cards5):
    """
    cards5: list of 5 tuples (rank_int, suit_char)
    Returns (rank_index, hand_label).
    """
    ranks = [r for r, s in cards5]
    suits = [s for r, s in cards5]

    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts = sorted(rank_counts.values(), reverse=True)
    flush = len(set(suits)) == 1
    straight = _is_straight_5(ranks)

    if straight and flush:
        return HAND_RANK_INDEX["Straight Flush"], "Straight Flush"
    if 4 in counts:
        return HAND_RANK_INDEX["Four of a Kind"], "Four of a Kind"
    if counts == [3, 2]:
        return HAND_RANK_INDEX["Full House"], "Full House"
    if flush:
        return HAND_RANK_INDEX["Flush"], "Flush"
    if straight:
        return HAND_RANK_INDEX["Straight"], "Straight"
    if 3 in counts:
        return HAND_RANK_INDEX["Three of a Kind"], "Three of a Kind"
    if counts == [2, 2, 1]:
        return HAND_RANK_INDEX["Two Pair"], "Two Pair"
    if counts == [2, 1, 1, 1]:
        return HAND_RANK_INDEX["One Pair"], "One Pair"
    return HAND_RANK_INDEX["High Card"], "High Card"


def classify_best_hand_from_codes(card_codes):
    """
    Given 5–7 cards, return best 5-card hand rank + label.
    """
    cards = _codes_to_cards_simple(card_codes)
    if len(cards) < 5:
        return HAND_RANK_INDEX["High Card"], "High Card"

    best_rank = -1
    best_label = "High Card"

    for combo in combinations(cards, 5):
        r_idx, label = _classify_5(combo)
        if r_idx > best_rank:
            best_rank = r_idx
            best_label = label

    return best_rank, best_label


def analyze_hand(hero_codes, board_codes):
    """
    Returns current hand + grouped draws with outs and probabilities.
    """
    all_known = hero_codes + board_codes
    current_rank, current_label = classify_best_hand_from_codes(all_known)

    unseen = [c for c in FULL_DECK_CODES if c not in all_known]
    board_len = len(board_codes)

    draws_by_group = {
        "straight": [],
        "flush": [],
        "full_house": [],
        "quads": [],
        "straight_flush": [],
        "other": [],
    }

    def group_for_label(label: str) -> str:
        if label == "Straight":
            return "straight"
        if label == "Flush":
            return "flush"
        if label == "Full House":
            return "full_house"
        if label == "Four of a Kind":
            return "quads"
        if label == "Straight Flush":
            return "straight_flush"
        return "other"

    cards_struct = _codes_to_cards_simple(all_known)
    rank_counts = {}
    for r, s in cards_struct:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    counts_sorted = sorted(rank_counts.values(), reverse=True)
    max_count = counts_sorted[0] if counts_sorted else 0
    pair_ranks = [r for r, c in rank_counts.items() if c == 2]
    trips_ranks = [r for r, c in rank_counts.items() if c == 3]

    def probs_from_outs(outs: int, board_len: int):
        remaining = 52 - len(all_known)
        res = {"prob_turn": None, "prob_river": None, "prob_by_river": None}

        if outs <= 0 or remaining <= 0:
            return res

        if board_len == 3:
            # Flop
            res["prob_turn"] = outs / remaining
            res["prob_river"] = outs / (remaining - 1)
            total_combos = math.comb(remaining, 2)
            miss_combos = math.comb(remaining - outs, 2) if remaining - outs >= 2 else 0
            res["prob_by_river"] = 1 - miss_combos / total_combos

        elif board_len == 4:
            # Turn
            res["prob_river"] = outs / remaining

        return res

    def rr_prob(combos: int):
        remaining = 52 - len(all_known)
        total_combos = math.comb(remaining, 2)
        if combos <= 0 or total_combos == 0:
            return None
        return combos / total_combos

    # One-card outs
    outs_counts = {}
    if board_len in (3, 4):
        for c in unseen:
            new_board = board_codes + [c]
            new_rank, new_label = classify_best_hand_from_codes(hero_codes + new_board)
            if new_rank >= HAND_RANK_INDEX["Straight"] and new_rank > current_rank:
                outs_counts[new_label] = outs_counts.get(new_label, 0) + 1

    # Runner-runner (backdoor) draws
    rr_counts = {}
    if board_len == 3:
        for c1, c2 in combinations(unseen, 2):
            final_board = board_codes + [c1, c2]
            final_rank, final_label = classify_best_hand_from_codes(hero_codes + final_board)

            if final_rank >= HAND_RANK_INDEX["Straight"] and final_rank > current_rank:
                r1, _ = classify_best_hand_from_codes(hero_codes + board_codes + [c1])
                r2, _ = classify_best_hand_from_codes(hero_codes + board_codes + [c2])

                if (
                    (r1 < HAND_RANK_INDEX["Straight"] or r1 <= current_rank)
                    and (r2 < HAND_RANK_INDEX["Straight"] or r2 <= current_rank)
                ):
                    rr_counts[final_label] = rr_counts.get(final_label, 0) + 1

    # Build one-card draws
    for label, outs in outs_counts.items():
        group = group_for_label(label)

        desc = label + " Draw"
        if label == "Straight":
            if outs >= 8:
                desc = "Open-Ended Straight Draw"
            elif outs == 4:
                desc = "Gutshot Straight Draw"
            elif outs > 4:
                desc = "Double Gutshot / Mixed Straight Draw"
        elif label == "Flush":
            desc = "Flush Draw"
        elif label == "Full House":
            if max_count >= 3:
                desc = "Full House Draw (Set → Full House)"
            elif len(pair_ranks) >= 2:
                desc = "Full House Draw (Two Pair → Full House)"
            elif len(pair_ranks) == 1:
                desc = "Full House Draw (Pair → Full House)"
            else:
                desc = "Full House Draw"
        elif label == "Four of a Kind":
            if max_count >= 3:
                desc = "Quads Draw (Set → Four of a Kind)"
            elif len(pair_ranks) == 1:
                desc = "Quads Draw (Pair → Four of a Kind)"
            else:
                desc = "Quads Draw"
        elif label == "Straight Flush":
            if outs <= 2:
                desc = "Straight Flush Draw (Very Strong)"
            else:
                desc = "Straight Flush Draw"

        probs = probs_from_outs(outs, board_len)

        draws_by_group[group].append(
            {
                "improves_to": label,
                "description": desc,
                "outs": outs,
                "prob_turn": probs["prob_turn"],
                "prob_river": probs["prob_river"],
                "prob_by_river": probs["prob_by_river"],
                "runner_runner_combos": 0,
                "runner_runner_prob": None,
            }
        )

    # Runner-runner entries
    for label, combos in rr_counts.items():
        group = group_for_label(label)

        if label == "Straight":
            desc = "Runner-runner Straight"
        elif label == "Flush":
            desc = "Runner-runner Flush"
        elif label == "Straight Flush":
            desc = "Runner-runner Straight Flush"
        elif label == "Full House":
            if max_count >= 3:
                desc = "Runner-runner Full House (Set → Full House)"
            elif len(pair_ranks) >= 2:
                desc = "Runner-runner Full House (Two Pair → Full House)"
            elif len(pair_ranks) == 1:
                desc = "Runner-runner Full House (Pair → Full House)"
            else:
                desc = "Runner-runner Full House"
        elif label == "Four of a Kind":
            desc = "Runner-runner Quads"
        else:
            desc = f"Runner-runner {label}"

        prob = rr_prob(combos)

        draws_by_group[group].append(
            {
                "improves_to": label,
                "description": desc,
                "outs": 0,
                "prob_turn": None,
                "prob_river": None,
                "prob_by_river": None,
                "runner_runner_combos": combos,
                "runner_runner_prob": prob,
            }
        )

    return {
        "hand_rank": current_rank,
        "hand_label": current_label,
        "draws": draws_by_group,
    }


# -------------------------------
# Board texture + bet sizing (aggressor)
# -------------------------------
def _board_texture_score(board_codes):
    """
    Score in [0,1]: higher = wetter / more dynamic.
    """
    cards = _codes_to_cards_simple(board_codes)
    if len(cards) < 3:
        return 0.5

    ranks = sorted({r for r, s in cards})
    suits = [s for r, s in cards]

    # Straight density
    if len(ranks) >= 3:
        diffs = [r2 - r1 for r1, r2 in zip(ranks, ranks[1:])]
        max_gap = max(diffs) if diffs else 4
        straight_density = max(0.0, min(1.0, (5 - max_gap) / 5))
    else:
        straight_density = 0.2

    # Flush density
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values())
    flush_density = max_suit / len(cards)

    # High cardness
    high_ranks = sum(1 for r in ranks if r >= 10)
    highness = high_ranks / len(ranks)

    # Paired penalty
    rank_counts = {}
    for r, s in cards:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    paired = any(c >= 2 for c in rank_counts.values())
    pairedness_penalty = 0.3 if paired else 0.0

    score = (
        straight_density * 0.4 +
        flush_density * 0.4 +
        highness * 0.15 -
        pairedness_penalty
    )
    return max(0.0, min(1.0, score))


def compute_bet_sizing(hero_eq, pot, street, board_codes, opponents):
    """
    Solver-style bet sizing when *we* are the aggressor (no bet in front).
    """
    try:
        eq = float(hero_eq)
    except (TypeError, ValueError):
        eq = 0.5

    try:
        pot_val = float(pot) if pot is not None else 0.0
    except (TypeError, ValueError):
        pot_val = 0.0

    street = (street or "").lower()
    if street not in {"flop", "turn", "river"}:
        street = "flop"

    has_real_pot = pot_val > 0
    pot_for_calc = pot_val if has_real_pot else 1.0

    texture = _board_texture_score(board_codes)
    opp = opponents or 1
    opp_factor = max(0.7, min(1.6, 1.0 + 0.15 * (opp - 1)))

    # Base sizes by street
    if street == "flop":
        base = 0.33
    elif street == "turn":
        base = 0.5
    else:
        base = 0.7

    # Equity band
    if eq < 0.35:
        equity_band = "low"
        eq_mult = 0.0
    elif eq < 0.50:
        equity_band = "bluff"
        eq_mult = 0.5
    elif eq < 0.65:
        equity_band = "medium"
        eq_mult = 1.0
    elif eq < 0.80:
        equity_band = "strong"
        eq_mult = 1.4
    else:
        equity_band = "nutty"
        eq_mult = 1.8

    texture_factor = 0.7 + 0.8 * texture

    raw_percent = base * eq_mult * texture_factor * opp_factor
    raw_percent = max(0.0, min(1.5, raw_percent))

    if equity_band == "low":
        recommended = {
            "action": "check-fold",
            "percent_of_pot": None,
            "amount": None,
            "label": "Low equity vs range — checking/folding is safest."
        }
        options = [
            {
                "action": "check",
                "percent_of_pot": None,
                "amount": None,
                "description": "Check / give up most of the time."
            }
        ]
        return {"recommended": recommended, "options": options}

    size_buckets = [0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25]
    best_size = min(size_buckets, key=lambda s: abs(s - raw_percent))

    options = []
    for size in size_buckets:
        amount = pot_for_calc * size if has_real_pot else None

        if size <= 0.3:
            desc = "Small range c-bet / probe"
        elif size <= 0.55:
            desc = "Medium value / protection bet"
        elif size <= 0.8:
            desc = "Large bet for strong value or big bluff"
        else:
            desc = "Overbet to polarize vs capped ranges"

        options.append(
            {
                "action": "bet",
                "percent_of_pot": size,
                "amount": amount,
                "description": desc,
            }
        )

    # Also include check as an option
    options.insert(
        0,
        {
            "action": "check",
            "percent_of_pot": None,
            "amount": None,
            "description": "Check to control pot / realize equity.",
        },
    )

    rec_amount = pot_for_calc * best_size if has_real_pot else None

    if equity_band == "bluff":
        label = "Low-to-medium equity — stab with a smaller size when fold equity is high."
    elif equity_band == "medium":
        label = "Medium equity — mix between checking and value/protection bets."
    elif equity_band == "strong":
        label = "Strong hand — lean into bigger sizing on dynamic boards."
    else:
        label = "Very strong hand — build a big pot, especially on wet textures."

    recommended = {
        "action": "bet",
        "percent_of_pot": best_size,
        "amount": rec_amount,
        "label": label,
    }

    return {
        "recommended": recommended,
        "options": options,
    }


# -------------------------------
# Villain action parsing + EV vs aggression
# -------------------------------
def parse_villain_action(text: str, pot_hint: float | None = None) -> dict:
    """
    Multi-action natural language parser for villain action.

    Supports full histories like:
      "sb checks, bb bets 30 into 45, hero calls, villain jams 180"

    We:
      - split on ',' and ';'
      - parse each clause
      - return the *last* relevant action (bet/raise/jam/check/call).
    """
    if not text:
        return {"type": "none", "amount": None, "pot_from_text": None, "raw": ""}

    raw_full = text

    def _parse_clause(clause: str, pot_hint_local: float | None):
        c_raw = clause
        t = clause.strip().lower()
        if not t:
            return {"type": "none", "amount": None, "pot_from_text": None, "raw": c_raw}

        # Action type
        if any(word in t for word in ["jam", "jams", "shove", "shoves", "all in", "all-in"]):
            act_type = "jam"
        elif "check" in t:
            act_type = "check"
        elif "raise" in t or "raises" in t:
            act_type = "raise"
        elif "bet" in t or "bets" in t:
            act_type = "bet"
        elif "call" in t or "calls" in t:
            act_type = "call"
        else:
            act_type = "unknown"

        # Numbers in this clause
        nums = re.findall(r"\d+(?:\.\d+)?", t)
        bet_amount = None
        pot_from_text = None

        if "into" in t and len(nums) >= 2:
            bet_amount = float(nums[-2])
            pot_from_text = float(nums[-1])
        elif len(nums) >= 2:
            bet_amount = float(nums[-2])
            pot_from_text = float(nums[-1])
        elif len(nums) == 1:
            bet_amount = float(nums[0])
            pot_from_text = None

        if pot_from_text is None and pot_hint_local is not None:
            try:
                pot_from_text = float(pot_hint_local)
            except (TypeError, ValueError):
                pot_from_text = None

        return {
            "type": act_type,
            "amount": bet_amount,
            "pot_from_text": pot_from_text,
            "raw": c_raw,
        }

    clauses = re.split(r"[;,]", text)
    parsed_clauses = [_parse_clause(cl, pot_hint) for cl in clauses]

    final = None
    for pc in parsed_clauses:
        if pc["type"] not in {"none", "unknown"}:
            final = pc

    if final is None:
        return {"type": "unknown", "amount": None, "pot_from_text": None, "raw": raw_full}

    final["raw"] = raw_full
    return final


def equity_realization_factor(hero_eq: float,
                              hand_rank: int,
                              position: str | None,
                              street: str) -> float:
    """
    Equity realization model (how much of your equity you actually realize).
    """
    try:
        eq = float(hero_eq)
    except (TypeError, ValueError):
        eq = 0.5

    max_rank_idx = len(HAND_ORDER) - 1
    made_score = hand_rank / max_rank_idx if max_rank_idx > 0 else 0.0
    strength_score = 0.6 * eq + 0.4 * made_score
    strength_score = max(0.0, min(1.0, strength_score))

    pos = (position or "").lower()
    if pos in {"btn", "bu", "button", "co", "cutoff", "ip"}:
        position_factor = 0.9   # in position
    elif pos in {"sb", "bb", "utg", "ep", "oop"}:
        position_factor = -0.4  # out of position
    else:
        position_factor = 0.0   # neutral / unknown

    base = 0.55 + 0.35 * strength_score + 0.1 * position_factor

    street = (street or "").lower()
    if street == "flop":
        mult = 0.9
    elif street == "turn":
        mult = 0.96
    else:
        mult = 1.0  # river

    realization = base * mult
    return max(0.5, min(1.0, realization))


def preflop_equity(hero_cards, opponents=1, trials=50000):
    """
    Monte Carlo simulation of equity preflop vs random hands.
    """
    hero = [eval7.Card(c) for c in hero_cards]

    deck = list(eval7.Deck())
    for c in hero:
        deck.remove(c)

    wins = ties = total = 0

    for _ in range(trials):
        random.shuffle(deck)

        # deal villains
        villain_hands = []
        idx = 0
        for _ in range(opponents):
            villain_hands.append([deck[idx], deck[idx + 1]])
            idx += 2

        board = deck[idx:idx + 5]

        hero_score = eval7.evaluate(hero + board)

        best_villain_score = max(eval7.evaluate(v + board) for v in villain_hands)

        if hero_score > best_villain_score:
            wins += 1
        elif hero_score == best_villain_score:
            ties += 1

        total += 1

    return {
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "equity": (wins + ties * 0.5) / total,
    }


def preflop_equity_vs_range(hero_cards, villain_range, trials=30000):
    """
    Monte Carlo simulation of equity preflop vs a specific villain range.

    BUG-FIXED:
      - Handles empty villain_range safely
      - Avoids hero/villain card collisions
      - Avoids deck.remove(x) errors
    """
    # --- SAFETY CHECK FOR EMPTY RANGE ---
    if not villain_range:
        return {
            "win_rate": 0.5,
            "tie_rate": 0.0,
            "equity": 0.5,
            "reason": "⚠ Villain range empty — using neutral 50% equity",
        }

    hero = [eval7.Card(c) for c in hero_cards]
    hero_strs = list(hero_cards)  # ['As','Kd'] etc.
    hero_set = set(hero_strs)

    # Map combos to actual card lists (AhKh, AdKh, AcKh, etc)
    def expand_combo(combo):
        r1, r2 = combo[0], combo[1]
        suited = "s" in combo
        offsuit = "o" in combo
        pairs = r1 == r2

        results = []
        for s1 in SUITS:
            for s2 in SUITS:
                if pairs and s1 >= s2:
                    continue
                if suited and s1 != s2:
                    continue
                if offsuit and s1 == s2:
                    continue
                if not suited and not offsuit and not pairs:
                    continue

                c1 = f"{r1}{s1}"
                c2 = f"{r2}{s2}"

                # *** AVOID HERO CARD COLLISIONS ***
                if c1 in hero_set or c2 in hero_set:
                    continue

                results.append(c1 + c2)
        return results

    # Build weighted list of actual card combos
    expanded = []
    for c in villain_range:
        expanded += expand_combo(c)

    # If somehow all combos overlapped with hero, fall back to neutral
    if not expanded:
        return {
            "win_rate": 0.5,
            "tie_rate": 0.0,
            "equity": 0.5,
            "reason": "⚠ All villain combos collided with hero — using neutral equity",
        }

    wins = ties = total = 0

    for _ in range(trials):
        deck = list(eval7.Deck())

        # remove hero cards
        for c in hero:
            if c in deck:
                deck.remove(c)

        # pick one random combo from villain range
        raw = random.choice(expanded)
        vc1, vc2 = raw[:2], raw[2:]
        villain = [eval7.Card(vc1), eval7.Card(vc2)]

        # If villain cards somehow overlap with hero (just in case), skip this trial
        if villain[0] in hero or villain[1] in hero:
            continue

        # remove villain cards if present
        if villain[0] in deck:
            deck.remove(villain[0])
        else:
            # collision or anomaly — skip this trial
            continue

        if villain[1] in deck:
            deck.remove(villain[1])
        else:
            # collision or anomaly — skip this trial
            continue

        # deal board
        board = deck[:5]

        hero_score = eval7.evaluate(hero + board)
        vill_score = eval7.evaluate(villain + board)

        if hero_score > vill_score:
            wins += 1
        elif hero_score == vill_score:
            ties += 1

        total += 1

    if total == 0:
        # Extreme edge case fallback
        return {
            "win_rate": 0.5,
            "tie_rate": 0.0,
            "equity": 0.5,
            "reason": "⚠ No valid trials executed — using neutral equity",
        }

    return {
        "win_rate": wins / total,
        "tie_rate": ties / total,
        "equity": (wins + ties * 0.5) / total,
    }


def evaluate_facing_jam(hero_eq: float,
                        hand_rank: int,
                        pot: float,
                        jam_size: float,
                        street: str,
                        position: str | None) -> dict:
    """
    EV model when facing an all-in (villain jam).
    """
    try:
        pot_val = float(pot)
    except (TypeError, ValueError):
        pot_val = 0.0

    try:
        call_val = float(jam_size)
    except (TypeError, ValueError):
        call_val = 0.0

    if pot_val <= 0 or call_val <= 0:
        action = "call" if hero_eq >= 0.5 else "fold"
        label = "No sizing info; using raw equity threshold."
        return {
            "action": action,
            "ev_call": None,
            "ev_fold": 0.0,
            "effective_equity": hero_eq,
            "label": label,
        }

    realization = equity_realization_factor(hero_eq, hand_rank, position, street)
    effective_eq = max(0.0, min(1.0, hero_eq * realization))

    total_pot_if_call = pot_val + call_val
    ev_call = effective_eq * total_pot_if_call - (1.0 - effective_eq) * call_val
    ev_fold = 0.0

    if ev_call > ev_fold:
        action = "call"
        label = f"Facing jam: EV(call) ≈ {ev_call:.2f} vs fold {ev_fold:.2f} (effective equity ~{effective_eq * 100:.1f}%)."
    else:
        action = "fold"
        label = f"Facing jam: EV(call) ≈ {ev_call:.2f} < fold {ev_fold:.2f} (effective equity ~{effective_eq * 100:.1f}%)."

    return {
        "action": action,
        "ev_call": ev_call,
        "ev_fold": ev_fold,
        "effective_equity": effective_eq,
        "label": label,
    }


def evaluate_facing_bet_or_raise(hero_eq: float,
                                 hand_rank: int,
                                 pot_before: float,
                                 bet_size: float,
                                 street: str,
                                 position: str | None,
                                 stack: float | None = None) -> dict:
    """
    EV model when facing a normal bet or raise (non-all-in).
    """
    try:
        pot_val = float(pot_before)
    except (TypeError, ValueError):
        pot_val = 0.0

    try:
        bet_val = float(bet_size)
    except (TypeError, ValueError):
        bet_val = 0.0

    if pot_val <= 0 or bet_val <= 0:
        return {
            "action": "unknown",
            "pot_odds": None,
            "effective_equity": hero_eq,
            "label": "Missing pot or bet size; cannot compute pot odds.",
            "ev_call": None,
            "ev_fold": 0.0,
            "raise_options": [],
        }

    pot_odds = bet_val / (pot_val + bet_val)

    realization = equity_realization_factor(hero_eq, hand_rank, position, street)
    effective_eq = max(0.0, min(1.0, hero_eq * realization))

    total_pot_if_call = pot_val + bet_val
    ev_call = effective_eq * total_pot_if_call - (1.0 - effective_eq) * bet_val
    ev_fold = 0.0

    if effective_eq > pot_odds:
        base_action = "call"
        label = (
            f"Facing bet: pot odds {pot_odds * 100:.1f}% vs effective equity {effective_eq * 100:.1f}% — calling is +EV."
        )
    else:
        base_action = "fold"
        label = (
            f"Facing bet: pot odds {pot_odds * 100:.1f}% > effective equity {effective_eq * 100:.1f}% — folding is safer."
        )

    raise_options = []
    max_raise = None
    try:
        if stack is not None:
            max_raise = float(stack) + bet_val  # rough: our stack + their bet
    except (TypeError, ValueError):
        max_raise = None

    for mult, tag in [(2.5, "small raise"), (3.5, "big raise")]:
        raise_to = bet_val * mult
        if max_raise is not None and raise_to > max_raise:
            continue
        raise_options.append(
            {
                "action": "raise",
                "amount": raise_to,
                "description": f"{tag} (~{mult:.1f}x vs bet)"
            }
        )

    if max_raise is not None:
        raise_options.append(
            {
                "action": "raise",
                "amount": max_raise,
                "description": "Jam (max pressure)"
            }
        )

    return {
        "action": base_action,
        "pot_odds": pot_odds,
        "effective_equity": effective_eq,
        "ev_call": ev_call,
        "ev_fold": ev_fold,
        "label": label,
        "raise_options": raise_options,
    }


# ----------------------------------------------------
# PRE-FLOP LOGIC
# ----------------------------------------------------
from .utils import parse_facing_bets
from .models import PreflopState


def to_combo(cards):
    """
    ['As','Kd'] -> 'AKo' / 'AKs' / 'AA' style string.
    """
    r1 = cards[0][0]
    r2 = cards[1][0]
    suited = cards[0][1] == cards[1][1]
    if r1 == r2:
        return r1 + r2
    return "".join(sorted([r1, r2], reverse=True)) + ("s" if suited else "o")


def _hero_pos_group(pos: str) -> str:
    pos = (pos or "").upper()
    if pos in {"UTG", "UTG+1", "EP"}:
        return "EP"
    if pos in {"LJ", "MP", "HJ"}:
        return "MP"
    if pos in {"CO", "BTN"}:
        return "LP"
    if pos in {"SB", "BB"}:
        return "BLINDS"
    return "UNKNOWN"


def _villain_pos_group(pos: str) -> str:
    pos = (pos or "").upper()
    if pos in {"UTG", "UTG+1", "EP"}:
        return "EP"
    if pos in {"LJ", "MP", "HJ"}:
        return "MP"
    if pos in {"CO"}:
        return "CO"
    if pos in {"BTN"}:
        return "BTN"
    if pos in {"SB", "BB"}:
        return "BLINDS"
    return "UNKNOWN"


def _map_villain_pos_to_range_key(vill_pos: str, ranges: dict) -> str:
    """
    Live-realistic tightening:
      - EP uses UTG range
      - MP uses something between UTG/HJ
      - CO uses CO/HJ
      - BTN uses CO/BTN
      - Blinds use their own if present, otherwise CO
    """
    vgroup = _villain_pos_group(vill_pos)

    # Normalize possible keys in ranges
    available = set(ranges.keys())

    # Helper to pick first existing key from candidates
    def pick(*candidates):
        for c in candidates:
            if c in available:
                return c
        # fallback
        if "UTG" in available:
            return "UTG"
        return next(iter(available))  # last resort

    if vgroup == "EP":
        return pick("UTG", "UTG+1", "EP")
    if vgroup == "MP":
        return pick("HJ", "MP", "UTG")
    if vgroup == "CO":
        return pick("CO", "HJ", "MP")
    if vgroup == "BTN":
        return pick("BTN", "CO")
    if vgroup == "BLINDS":
        return pick("SB", "BB", "CO")

    # Unknown → fallback to something reasonable
    return pick("CO", "HJ", "UTG")


def decide_preflop(state: PreflopState) -> dict:
    """
    Preflop advisor logic that accounts for action in front.
    Expected fields on state:
        hero_cards: ['As','Kd']
        position: 'BTN'
        facing_bets: dict like {"action": "raise", "aggressor": "UTG", "size": 3}
    """

    hero_cards = state.hero_cards
    hero = " ".join(hero_cards)
    pos = (state.position or "").upper()
    facing = state.facing_bets or {}

    combo = to_combo(hero_cards)

    action_type = facing.get("action")      # raise / 3bet / open / None
    aggressor = facing.get("aggressor")     # UTG / CO / BTN etc.
    size = facing.get("size")               # numeric raise size in bb

    # ------------------------------
    # CASE 0 — MISSING OR BAD INPUT
    # ------------------------------
    if not hero_cards or len(hero_cards) != 2:
        return {
            "action": "error",
            "reason": "Missing or invalid hero hand",
            "raw": facing
        }

    # ------------------------------
    # CASE 1 — OPEN RAISE (no prior action)
    # ------------------------------
    if not action_type:
        eq = preflop_equity(hero_cards, opponents=1)
        equity_pct = round(eq["equity"] * 100, 2)

        return {
            "action": "open-raise",
            "size": "2.5bb",
            "equity": {
                "win_rate": round(eq["win_rate"] * 100, 2),
                "tie_rate": round(eq["tie_rate"] * 100, 2),
                "total_equity": equity_pct,
            },
            "reason": f"Opening {hero} from {pos} — equity {equity_pct}%",
            "raw": facing
        }

    # Load full-range file once
    ranges = load_ranges("data/ranges/preflop_live_fullring_100bb.json")

    # ------------------------------
    # CASE 2 — FACING A STANDARD RAISE
    # ------------------------------
    if action_type == "raise":
        size = size or 3.0

        hero_pos_group = _hero_pos_group(pos)
        vill_pos_raw = (aggressor or "").upper()
        vill_range_key = _map_villain_pos_to_range_key(vill_pos_raw, ranges)

        vill_range_block = ranges.get(vill_range_key, {})
        vill_range = vill_range_block.get("open", [])

        # Fallback if still empty
        if not vill_range:
            vill_range_key = "UTG"
            vill_range = ranges.get("UTG", {}).get("open", [])

        # Compute equity vs that range
        eq = preflop_equity_vs_range(hero_cards, vill_range)
        equity = float(eq["equity"])
        equity_pct = round(equity * 100, 2)

        win_pct = round(eq.get("win_rate", 0.0) * 100, 2)
        tie_pct = round(eq.get("tie_rate", 0.0) * 100, 2)

        vill_group = _villain_pos_group(vill_pos_raw)

        # -----------------------------
        # POS-ADJUSTED THRESHOLDS
        # -----------------------------
        # Baseline thresholds (equity) for continuing vs open
        if vill_group == "EP":
            min_call_eq = 0.50
            min_3bet_eq = 0.56
        elif vill_group in {"MP"}:
            min_call_eq = 0.47
            min_3bet_eq = 0.54
        elif vill_group in {"CO"}:
            min_call_eq = 0.46
            min_3bet_eq = 0.53
        elif vill_group in {"BTN", "BLINDS"}:
            min_call_eq = 0.45
            min_3bet_eq = 0.52
        else:
            min_call_eq = 0.48
            min_3bet_eq = 0.55

        # IP vs OOP tweak
        hero_ip = hero_pos_group in {"LP"} and vill_group in {"EP", "MP", "CO"}
        if hero_ip:
            min_call_eq -= 0.01  # loosen IP
            min_3bet_eq -= 0.01
        elif hero_pos_group == "BLINDS":
            min_call_eq += 0.01  # tighten OOP in blinds
            min_3bet_eq += 0.01

        # -----------------------------
        # HAND-CLASS-BASED DECISION
        # -----------------------------
        equity_info = {
            "win_rate": win_pct,
            "tie_rate": tie_pct,
            "total_equity": equity_pct,
            "vs_range": f"{vill_range_key} open ({vill_group})"
        }

        # PREMIUM — mostly 3-bet for value
        if combo in PREMIUM_HANDS:
            return {
                "action": "3-bet",
                "size": f"{size * 3:.1f}bb",
                "reason": (
                    f"Premium hand {combo} vs {vill_pos_raw} open "
                    f"({equity_pct}% range-adjusted equity). Value 3-bet."
                ),
                "equity": equity_info
            }

        # STRONG — mix between call and 3-bet depending on position + equity
        if combo in STRONG_HANDS:
            if equity >= min_3bet_eq and hero_ip:
                # Aggressive vs late opens IP
                return {
                    "action": "3-bet",
                    "size": f"{size * 3.0:.1f}bb",
                    "reason": (
                        f"Strong hand {combo} IP vs {vill_pos_raw} open — "
                        f"{equity_pct}% equity and in position → mixed but prefer 3-bet."
                    ),
                    "equity": equity_info
                }
            elif equity >= min_call_eq:
                return {
                    "action": "call",
                    "reason": (
                        f"Strong playable hand {combo} vs {vill_pos_raw} open — "
                        f"{equity_pct}% equity, defend by calling."
                    ),
                    "equity": equity_info
                }
            else:
                return {
                    "action": "fold",
                    "reason": (
                        f"{combo} is strong but under-realizing vs tight {vill_pos_raw} open — "
                        f"{equity_pct}% equity below live threshold."
                    ),
                    "equity": equity_info
                }

        # MEDIUM — more sensitive to position + opener
        if combo in MEDIUM_HANDS:
            if equity >= min_call_eq and hero_ip and vill_group in {"CO", "BTN", "MP"}:
                return {
                    "action": "call",
                    "reason": (
                        f"Medium strength {combo} IP vs {vill_pos_raw} open — "
                        f"{equity_pct}% equity, okay to peel a flop."
                    ),
                    "equity": equity_info
                }
            else:
                return {
                    "action": "fold",
                    "reason": (
                        f"{combo} is marginal vs {vill_pos_raw} open — "
                        f"{equity_pct}% equity below positional defend threshold."
                    ),
                    "equity": equity_info
                }

        # WEAK — almost always fold vs raise
        if combo in WEAK_HANDS or combo in TRASH_HANDS:
            return {
                "action": "fold",
                "reason": (
                    f"Weak hand {combo} vs {vill_pos_raw} open — "
                    f"{equity_pct}% equity, fold preflop."
                ),
                "equity": equity_info
            }

        # Default for anything else — use pure equity threshold
        if equity >= min_call_eq:
            return {
                "action": "call",
                "reason": (
                    f"{combo} vs {vill_pos_raw} open — "
                    f"{equity_pct}% range-adjusted equity justifies a defend."
                ),
                "equity": equity_info
            }

        return {
            "action": "fold",
            "reason": (
                f"{combo} too weak vs {vill_pos_raw} open — "
                f"{equity_pct}% equity below defend threshold."
            ),
            "equity": equity_info
        }

    # ------------------------------
    # CASE 3 — FACING A 3-BET
    # ------------------------------
    if action_type == "3bet":
        # Placeholder logic — could be extended with range vs 3-bet
        return {
            "action": "call" if combo in PREMIUM_HANDS or combo in STRONG_HANDS else "fold",
            "reason": f"Simplified 3-bet defense vs {aggressor} with {combo}",
            "raw": facing
        }

    # ------------------------------
    # CASE 4 — OTHER ACTION TYPES (limps, etc.)
    # ------------------------------
    if action_type == "open":
        return {
            "action": "raise",
            "size": "3bb",
            "reason": f"Punish open limp from {aggressor}",
            "raw": facing
        }

    # ------------------------------
    # FALLBACK — shouldn't happen normally
    # ------------------------------
    return {
        "action": "unknown",
        "reason": "Action detected but not categorized",
        "raw": facing
    }


# ----------------------------------------------------
# POST-FLOP LOGIC
# ----------------------------------------------------
# ----------------------------------------------------
# POST-FLOP LOGIC
# ----------------------------------------------------
def decide_postflop(state):

    hero = state.hero_cards
    board = state.board
    opponents = getattr(state, "opponents", None) or 1
    facing = getattr(state, "facing_action", {}) or {}
    street = getattr(state, "street", "flop")

    # Convert cards
    hero_cards = [eval7.Card(c) for c in hero]
    board_cards = [eval7.Card(c) for c in board]

    # Equity
    eqh, eqv, tie, dist = equity_vs_random_opponents(hero_cards, board_cards)
    eq = eqh

    # Hand + draws
    hand_info = analyze_hand(hero, board)
    hand_label = hand_info["hand_label"]
    hand_rank = hand_info["hand_rank"]
    draws = hand_info["draws"]

    action_type = facing.get("type")
    amount = facing.get("amount")

    # Base return data
    result = {
        "hero_equity": round(eqh, 2),
        "villain_equity": round(eqv, 2),
        "tie": round(tie, 2),
        "equity_raw": eq,
        "board": board,
        "hand": hero,
        "hand_label": hand_label,
        "hand_rank": hand_rank,
        "draws": draws,
    }

    # -------------------------------------------------
    # Compute pot_before once, safely
    # -------------------------------------------------
    raw_pot = getattr(state, "pot", None)
    pot_before = None

    if raw_pot not in (None, ""):
        try:
            pot_before = float(raw_pot)
        except (TypeError, ValueError):
            pot_before = None

    if pot_before is None:
        pot_hint = facing.get("pot_from_text")
        if pot_hint not in (None, ""):
            try:
                pot_before = float(pot_hint)
            except (TypeError, ValueError):
                pot_before = 0.0
        else:
            pot_before = 0.0

    # -------------------------------------------------
    # CASE 1: FACING ACTION (bet / raise / jam)
    # -------------------------------------------------
    if action_type in ("bet", "raise") and amount is not None:
        # Normal bet / raise
        call_eval = evaluate_facing_bet_or_raise(
            hero_eq=eq,
            hand_rank=hand_rank,
            pot_before=pot_before,
            bet_size=amount,
            street=street,
            position=getattr(state, "position", None),
            stack=getattr(state, "stack", None),
        )

        return {
            **result,
            "action": call_eval["action"],
            "reason": call_eval["label"],
            "recommended": call_eval,
            "facing_aggression": facing,
        }

    if action_type == "jam" and amount is not None:
        jam_eval = evaluate_facing_jam(
            hero_eq=eq,
            hand_rank=hand_rank,
            pot=pot_before,
            jam_size=amount,
            street=street,
            position=getattr(state, "position", None),
        )

        return {
            **result,
            "action": jam_eval["action"],
            "reason": jam_eval["label"],
            "recommended": jam_eval,
            "facing_aggression": facing,
        }

    # -------------------------------------------------
    # CASE 2: NO BET → WE ARE THE AGGRESSOR
    # -------------------------------------------------
    pot_for_bet = pot_before  # use same resolved pot

    bet_eval = compute_bet_sizing(
        hero_eq=eq,
        pot=pot_for_bet,
        street=street,
        board_codes=board,
        opponents=opponents,
    )

    return {
        **result,
        "action": bet_eval["recommended"]["action"],
        "bet_options": bet_eval["options"],
        "recommended": bet_eval["recommended"],
        "reason": bet_eval["recommended"]["label"],
        "facing_aggression": None,
    }
