


from typing import Dict, Tuple

def betting_decision(odds: Dict[str, float],
                     probs: Dict[str, float],
                     bankroll: float,
                     kelly_fraction: float = 0.5,
                     min_edge: float = 0.02) -> Dict[str, Tuple[bool, float, float]]:
    """
    Decide whether to bet based on your model probabilities vs bookmaker odds.
    
    Parameters
    ----------
    odds : dict
        Decimal odds for outcomes, e.g. {"home": 2.40, "draw": 3.30, "away": 3.10}
    probs : dict
        Your model probabilities, e.g. {"home": 0.46, "draw": 0.26, "away": 0.28}
    bankroll : float
        Current bankroll size (e.g. 1000 for £1000).
    kelly_fraction : float, optional
        Fraction of Kelly stake to use (default 0.5 = half Kelly).
    min_edge : float, optional
        Minimum edge threshold to consider betting (default 0.02 = 2%).
    
    Returns
    -------
    dict
        Keys are outcomes ("home", "draw", "away").
        Values are tuples: (should_bet, stake_amount, expected_value_per_unit).
    """
    decisions = {}
    for outcome, O in odds.items():
        p = probs[outcome]
        
        # Expected value per £1 stake
        EV = p * O - 1
        
        # Kelly fraction of bankroll
        kelly = (p * O - 1) / (O - 1) if O > 1 else 0
        
        # Apply fractional Kelly
        stake_fraction = max(0, kelly_fraction * kelly)
        stake_amount = stake_fraction * bankroll
        
        # Decision: bet only if EV > min_edge
        should_bet = EV > min_edge
        
        decisions[outcome] = (should_bet, round(stake_amount, 2), round(EV, 4))
    
    return decisions





for i in range(0, len(fixtures[0])):
	fixture = fixtures[0][i]
	print("#-------------------------------#")
	print(fixture)
	pred = meta['predict_fixture'](fixture[0],fixture[1])
	print(f"{fixture[0]} win: {pred['probabilities']['home_win']}")
	print(f"Draw: {pred['probabilities']['draw']}")
	print(f"{fixture[1]} win: {pred['probabilities']['away_win']}")
	df_matrix = pd.DataFrame(pred['probabilities']['matrix'])
	df_matrix.index.name = f"{fixture[0]}_goals"
	df_matrix.columns.name = f"{fixture[1]}_goals"
	print(df_matrix.loc[0:2, 0:2])




# Example usage
odds = {"home": 2.55, "draw": 3.3, "away": 2.8}
probs = {"home": 0.4696000658377082, "draw": 0.34480881906849103, "away": 0.18549793838165932}
bankroll = 10

result = betting_decision(odds, probs, bankroll)
print(result)


import numpy as np
from typing import List, Dict

def betting_plan_from_lists(prob_list: List[Dict[str, np.float64]],
                            odds_list: List[Dict[str, float]],
                            bankroll: float,
                            kelly_fraction: float = 0.5,
                            min_edge: float = 0.02) -> List[Dict[str, object]]:
    """
    Generate betting decisions for multiple fixtures given lists of dicts.
    
    Parameters
    ----------
    prob_list : list of dicts
        Your model probabilities for each fixture.
        Example: [{"home": 0.52, "draw": 0.38, "away": 0.09}, ...]
    odds_list : list of dicts
        Bookmaker odds for each fixture in same order.
        Example: [{"home": 2.40, "draw": 3.30, "away": 3.10}, ...]
    bankroll : float
        Total bankroll to allocate.
    kelly_fraction : float
        Fraction of Kelly stake to use (default 0.5).
    min_edge : float
        Minimum EV threshold to consider betting.
    
    Returns
    -------
    list of dicts
        Each dict contains fixture index, outcome, prob, odds, EV, stake, and bet decision.
    """
    results = []
    n_fixtures = len(prob_list)
    fixture_bankroll = bankroll / n_fixtures  # equal split across fixtures
    
    for idx, (probs, odds) in enumerate(zip(prob_list, odds_list)):
        for outcome in ["home", "draw", "away"]:
            p = float(probs[outcome])
            O = float(odds[outcome])
            
            EV = p * O - 1
            kelly = (p * O - 1) / (O - 1) if O > 1 else 0
            stake_fraction = max(0, kelly_fraction * kelly)
            stake_amount = stake_fraction * fixture_bankroll
            
            bet = EV > min_edge
            results.append({
                "fixture": idx,
                "outcome": outcome,
                "prob": round(p, 3),
                "odds": O,
                "EV": round(EV, 4),
                "stake": round(stake_amount, 2),
                "bet": bet
            })
    
    return results




prob_list = []

for i in range(0, len(fixtures[0])):
	fixture = fixtures[0][i]
	print("#-------------------------------#")
	print(fixture)
	pred = meta['predict_fixture'](fixture[0],fixture[1])
	print(f"{fixture[0]} win: {pred['probabilities']['home_win']}")
	print(f"Draw: {pred['probabilities']['draw']}")
	print(f"{fixture[1]} win: {pred['probabilities']['away_win']}")
	prob_list.append({"home": pred['probabilities']['home_win'], "draw": pred['probabilities']['draw'], "away": pred['probabilities']['away_win']},)


odds_list = [
    {'home': 2.60, 'draw': 3.5, 'away': 2.5},
    {'home': 1.47, 'draw': 4.4, 'away': 6.5},
    {'home': 1.61, 'draw': 4.00, 'away': 5.00},
    {'home': 1.20, 'draw': 7.00, 'away': 11.00},
    {'home': 3.5, 'draw': 3.4, 'away': 2.05},
    {'home': 3.4, 'draw': 3.6, 'away': 2.05},
    {'home': 5.5, 'draw': 3.8, 'away': 1.61},
    {'home': 2.70, 'draw': 3.25, 'away': 2.6},
    {'home': 2.05, 'draw': 3.6, 'away': 3.3},
    {'home': 2.2, 'draw': 3.3, 'away': 3.2}
]



plan = betting_plan_from_lists(prob_list, odds_list, bankroll=20, min_edge = 0.5)
for row in plan:
    if row['bet'] == True:
    	print(fixtures[0][row['fixture']])
    	print(row)

numbers = [0.24, 0.09, 0.49, 0.21, 0.4, 0.48]
target_sum = 20
current_sum = sum(numbers)
# Scale each number so the new list sums to target_sum
scaled = [num * target_sum / current_sum for num in numbers]















import numpy as np
from typing import List, Dict, Tuple, Optional

OutcomeDict = Dict[str, float]  # {"home": O_H, "draw": O_D, "away": O_A}

def best_odds_across_books(odds_books: List[OutcomeDict]) -> Tuple[OutcomeDict, Dict[str, Tuple[float, int]]]:
    """
    Select the best odds per outcome across multiple bookmakers.
    
    Returns:
        best_odds: dict with max odds per outcome.
        provenance: dict outcome -> (odds, book_index) for tracking where to place the bet.
    """
    outcomes = ["home", "draw", "away"]
    best_odds = {}
    provenance = {}
    for outcome in outcomes:
        best_val = -np.inf
        best_idx = -1
        for i, book in enumerate(odds_books):
            val = float(book[outcome])
            if val > best_val:
                best_val, best_idx = val, i
        best_odds[outcome] = best_val
        provenance[outcome] = (best_val, best_idx)
    return best_odds, provenance


def ev_and_kelly(p: float, O: float) -> Tuple[float, float]:
    """
    Compute expected value per unit stake and Kelly fraction for decimal odds.
    Returns (EV, kelly_fraction_raw).
    """
    EV = p * O - 1
    kelly = (p * O - 1) / (O - 1) if O > 1 else 0.0
    return EV, kelly


def detect_arbitrage_3way(best_odds: OutcomeDict) -> Tuple[bool, float]:
    """
    Check if cross-book arbitrage exists in a 3-way market using best odds.
    Returns (is_arb, overround), where overround = sum(1/O_i).
    Arbitrage exists if overround < 1.
    """
    S = sum(1.0 / float(best_odds[k]) for k in ["home", "draw", "away"])
    return (S < 1.0), S


def dutch_stakes_for_arbitrage(best_odds: OutcomeDict, total_stake: float) -> Dict[str, float]:
    """
    Compute dutching stakes to lock profit across home/draw/away given best odds.
    Assumes arbitrage condition is met.
    """
    inv = {k: 1.0 / float(best_odds[k]) for k in ["home", "draw", "away"]}
    S = sum(inv.values())
    return {k: round(total_stake * inv[k] / S, 2) for k in inv}


def betting_plan_multi_books(
    prob_list: List[Dict[str, float]],
    odds_lists_per_fixture: List[List[OutcomeDict]],
    bankroll: float,
    kelly_fraction: float = 0.5,
    min_edge: float = 0.02,
    max_fixture_exposure: float = 0.05,
    hedge_positive_ev_only: bool = True
) -> Dict[str, object]:
    """
    Build a betting plan over multiple fixtures with multiple bookmakers per fixture.
    
    Inputs:
        prob_list: list of dicts with probabilities per fixture {"home": pH, "draw": pD, "away": pA}.
        odds_lists_per_fixture: list of lists; each inner list contains bookmaker odds dicts for that fixture.
        bankroll: total bankroll.
        kelly_fraction: fraction of Kelly to use.
        min_edge: minimum EV threshold to place a bet.
        max_fixture_exposure: cap of bankroll per fixture (e.g., 0.05 = 5%).
        hedge_positive_ev_only: when multiple outcomes have EV > min_edge, spread stakes across them;
                                otherwise do not hedge into negative EV.
    
    Returns:
        dict with:
            - "selections": list of dicts for value bets with recommended stakes and which book to use.
            - "arbitrage": list of dicts for fixtures where guaranteed profit dutching is possible.
            - "summary": totals.
    """
    plan = {"selections": [], "arbitrage": [], "summary": {}}
    n_fixtures = len(prob_list)
    fixture_cap = bankroll * max_fixture_exposure

    total_recommended_stakes = 0.0
    total_expected_value = 0.0

    for idx, (probs, books) in enumerate(zip(prob_list, odds_lists_per_fixture)):
        # 1) Best odds across books
        best_odds, provenance = best_odds_across_books(books)

        # 2) Arbitrage detection
        is_arb, overround = detect_arbitrage_3way(best_odds)
        arb_entry: Optional[Dict[str, object]] = None
        if is_arb:
            # Suggest a conservative arb stake (e.g., min of fixture cap and 1% of bankroll)
            arb_total = min(fixture_cap, bankroll * 0.01)
            dutch = dutch_stakes_for_arbitrage(best_odds, arb_total)
            guaranteed_profit = round(arb_total * (1.0 / overround - 1.0), 2)
            arb_entry = {
                "fixture": idx,
                "best_odds": best_odds,
                "overround": round(overround, 4),
                "total_stake": arb_total,
                "stakes": dutch,
                "guaranteed_profit": guaranteed_profit,
                "books": {k: provenance[k][1] for k in ["home", "draw", "away"]}
            }
            plan["arbitrage"].append(arb_entry)

        # 3) Value bets using model probabilities (per outcome)
        outcomes_ev = {}
        for outcome in ["home", "draw", "away"]:
            p = float(probs[outcome])
            O = float(best_odds[outcome])
            EV, k = ev_and_kelly(p, O)
            outcomes_ev[outcome] = {"p": p, "O": O, "EV": EV, "kelly": k, "book": provenance[outcome][1]}

        # Filter to positive EV outcomes (above threshold)
        candidates = {o: d for o, d in outcomes_ev.items() if d["EV"] > min_edge}

        if not candidates:
            continue

        # 4) Hedge across multiple positive EV outcomes (optional)
        #    Allocate fixture exposure proportionally to EV (or Kelly) while respecting cap.
        #    This reduces variance without betting into negative EV.
        fixture_exposure = fixture_cap
        weights_source = {o: max(0.0, kelly_fraction * d["kelly"]) for o, d in candidates.items()}
        if sum(weights_source.values()) == 0:
            # Fallback to EV-based weights if Kelly is zero (e.g., tiny edges)
            weights_source = {o: d["EV"] for o, d in candidates.items()}

        total_weight = sum(weights_source.values())
        for outcome, d in candidates.items():
            w = weights_source[outcome] / total_weight if total_weight > 0 else 0.0
            stake = round(fixture_exposure * w, 2)
            total_recommended_stakes += stake
            total_expected_value += stake * d["EV"]
            plan["selections"].append({
                "fixture": idx,
                "outcome": outcome,
                "prob": round(d["p"], 4),
                "odds": d["O"],
                "EV": round(d["EV"], 4),
                "kelly_raw": round(d["kelly"], 4),
                "stake": stake,
                "book_index": d["book"]
            })

    plan["summary"] = {
        "fixtures_considered": n_fixtures,
        "arbitrage_count": len(plan["arbitrage"]),
        "total_recommended_stakes": round(total_recommended_stakes, 2),
        "expected_value_of_plan": round(total_expected_value, 2),
        "bankroll": bankroll,
        "fixture_exposure_cap": round(fixture_cap, 2)
    }
    return plan





prob_list = []

for i in range(0, len(fixture_1)):
	fixture = fixture_1[i]
	print("#-------------------------------#")
	print(fixture)
	pred = meta['predict_fixture'](fixture[0],fixture[1])
	print(f"{fixture[0]} win: {pred['probabilities']['home_win']}")
	print(f"Draw: {pred['probabilities']['draw']}")
	print(f"{fixture[1]} win: {pred['probabilities']['away_win']}")
	prob_list.append({"home": pred['probabilities']['home_win'], "draw": pred['probabilities']['draw'], "away": pred['probabilities']['away_win']},)

#1 ladbrookes
#2beeetfred
#univeet
odds_lists_per_fixture = [
 [
    {'home': 1.65, 'draw': 3.90, 'away': 5.00},
    {'home': 1.60, 'draw': 3.60, 'away': 5.25},
     {'home': 1.65, 'draw': 3.90, 'away': 5.00}
 ],
 [  {'home': 1.70, 'draw': 4.00, 'away': 4.40},
    {'home': 1.67, 'draw': 3.75, 'away': 4.20},
         {'home': 1.70, 'draw': 4.2, 'away': 4.35}
  ],
 [   {'home': 3.75, 'draw': 3.50, 'away': 1.95},
  {'home': 3.75, 'draw': 3.20, 'away': 1.90},
       {'home': 3.90, 'draw': 3.50, 'away': 1.95}
  ],
 [   {'home': 1.13, 'draw': 8.50, 'away': 17.00},
 {'home': 1.11, 'draw': 7.00, 'away': 17.00},
      {'home': 1.14, 'draw': 7.5, 'away': 19.00}
  ],
 [   {'home': 3.80, 'draw': 3.70, 'away': 1.90},
 {'home': 3.60, 'draw': 3.50, 'away': 1.83},
      {'home': 3.90, 'draw': 3.80, 'away': 1.85}
  ],
 [   {'home': 2.50, 'draw': 3.30, 'away': 2.75},
 {'home': 2.40, 'draw': 3.10, 'away': 2.75},
      {'home': 2.50, 'draw': 3.35, 'away': 2.75}
  ],
 [   {'home': 3.25, 'draw': 3.30, 'away': 2.20},
 {'home': 3.10, 'draw': 3.20, 'away': 2.15},
      {'home': 3.25, 'draw': 3.30, 'away': 2.20}
  ],
 [   {'home': 3.60, 'draw': 3.50, 'away': 2.00},
 {'home': 3.50, 'draw': 3.30, 'away': 1.95},
      {'home': 3.60, 'draw': 3.50, 'away': 2.00}
  ],
 [   {'home': 2.00, 'draw': 3.50, 'away': 3.70},
 {'home': 1.95, 'draw': 3.25, 'away': 3.5},
      {'home': 1.98, 'draw': 3.50, 'away': 3.75}
  ],
 [   {'home': 1.83, 'draw': 3.90, 'away': 3.80},
 {'home': 1.80, 'draw': 3.60, 'away': 3.75},
      {'home': 1.84, 'draw': 3.90, 'away': 3.80}
  ]
]


plan = betting_plan_multi_books(
    prob_list,
    odds_lists_per_fixture,
    bankroll=40,           # total bankroll
    kelly_fraction=0.5,       # fractional Kelly
    min_edge=0.05,            # require at least +5 tgggggg8y% EV
    max_fixture_exposure=0.15 # cap 15% per fixture
)

# Inspect plan["selections"] for value bets, plan["arbitrage"] for guaranteed plays, plan["summary"] for totals.

#0 ladbrookes
#1 beeetfred
#2 univeet
pd.DataFrame(plan["selections"]).sort_values('book_index')
fixture_1[0]