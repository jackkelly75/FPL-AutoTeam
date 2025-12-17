import requests, json
from pprint import pprint
import csv
import time
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

from datetime import datetime, timezone
BASE = "https://fantasy.premierleague.com/api"


# Set constraints (customize as needed)
constraints = {
    'GK': 1,
    'DEF': {'min': 3, 'max': 5},
    'MID': {'min': 2, 'max': 5},
    'FWD': {'min': 1, 'max': 3}
}
#----- Functions to get fixtures for gameweek
def fetch_json(url, session=None):
    s = session or requests.Session()
    r = s.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def build_team_map(bootstrap_json):
    # bootstrap_static has "teams" list with id and name
    return {t["id"]: t["name"] for t in bootstrap_json["teams"]}

def fixtures_for_event(event, session=None):
    fixtures = fetch_json("https://fantasy.premierleague.com/api/fixtures/", session=session)
    bootstrap = fetch_json("https://fantasy.premierleague.com/api/bootstrap-static/", session=session)
    id_to_name = build_team_map(bootstrap)

    # Filter fixtures for the requested gameweek (event)
    event_fixtures = [
        (id_to_name[f["team_h"]], id_to_name[f["team_a"]])
        for f in fixtures
        if f.get("event") == event
    ]
    return event_fixtures


# --------- function to pull the boostrap data
def get_fpl_data():
    base_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(base_url)
    data = response.json()
    players_data = pd.DataFrame(data['elements'])
    teams_data = pd.DataFrame(data['teams'])
    events_data = pd.DataFrame(data['events'])
    return players_data, teams_data, events_data

def session_with_retries():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "fpl-data-script/1.0"})
    return s

def get_bootstrap(session):
    url = f"{BASE}/bootstrap-static/"
    return session.get(url, timeout=30).json()

def get_player_history(session, player_id):
    url = f"{BASE}/element-summary/{player_id}/"
    return session.get(url, timeout=30).json()


def import_player_data(num_mins = 1):
	players_data, teams_data, events_data = get_fpl_data()
	s = session_with_retries()
	print("Downloading bootstrap-static (players list)...")
	bs = get_bootstrap(s)
	teams_map = {t['id']: t['name'] for t in bs['teams']}  # id -> full team name
	players = {p['id']: p for p in bs['elements']}
	#import weekly player data
	save_data = []
	print(f"Fetching history for {len(players)} players...")
	for pid in tqdm(sorted(players.keys())):
	    pmeta = players[pid]
	    try:
	        data = get_player_history(s, pid)
	    except Exception as e:
	        print(f"Error fetching {pid}: {e}; skipping.")
	        continue

	    for gw in data.get("history", []):
	        row = {
	            "player_id": pid,
	            "first_name": pmeta.get("first_name"),
	            "second_name": pmeta.get("second_name"),
	            "team_id": pmeta.get("team"),
	            "team_name": teams_map.get(pmeta.get("team")),     # readable string
	            "element_type": pmeta.get("element_type"),
	            "gameweek": gw.get("round"),
	            "kickoff_time": gw.get("kickoff_time"),
	            "minutes": gw.get("minutes"),
	            "G": gw.get("goals_scored"),
	            "xG": gw.get("expected_goals"),
	            "assists": gw.get("assists"),
	            "xassists": gw.get("expected_assists"),
	            "clean_sheets": gw.get("clean_sheets"),
	            "goals_conceded": gw.get("goals_conceded"),
	            "x_goals_conceded": gw.get("expected_goals_conceded"),
	            "own_goals": gw.get("own_goals"),
	            "clearances_blocks_interceptions": gw.get("clearances_blocks_interceptions"),
	            "defensive_contribution": gw.get("defensive_contribution"),
	            "penalties_saved": gw.get("penalties_saved"),
	            "penalties_missed": gw.get("penalties_missed"),
	            "yellow_cards": gw.get("yellow_cards"),
	            "red_cards": gw.get("red_cards"),
	            "saves": gw.get("saves"),
	            "bonus": gw.get("bonus"),
	            "bps": gw.get("bps"),
	            "influence": gw.get("influence"),
	            "creativity": gw.get("creativity"),
	            "threat": gw.get("threat"),
	            "ict_index": gw.get("ict_index"),
	            "total_points": gw.get("total_points"),
	            "was_home": gw.get("was_home"),
	            #"opponent_team": gw.get("opponent_team"),
		        "opponent_team_name": teams_map.get(gw.get("opponent_team"))
	        }
	        save_data.append(row)
	save_data = pd.DataFrame(save_data)
	save_data['G'] = save_data['G'].astype(float)
	save_data['xG'] = save_data['xG'].astype(float)
	save_data['assists'] = save_data['assists'].astype(float)
	save_data['xassists'] = save_data['xassists'].astype(float)
	players_data = players_data[['id', 'now_cost', 'web_name']]
	players_data = players_data.rename(columns={"id": "player_id"})
	save_data = save_data.merge(players_data, on = ['player_id'], how  = 'inner')
	#clean some data 
	save_data = save_data.rename(columns={"web_name": "player_name"})
	save_data = save_data.rename(columns={"round": "gameweek"})
	save_data = save_data.rename(columns={"team": "team_name"})
	save_data['now_cost'] = save_data['now_cost']/10
	#reduce down to players who have played at least num_mins this season
	save_data = save_data[save_data.groupby('player_name')['minutes'].transform('sum') >= 1]
	return save_data


from decimal import Decimal, ROUND_DOWN
def half_profit_round_down_to_0_1(profit):
    # use Decimal(str(...)) to preserve the human-readable value
    d = Decimal(str(profit)) / Decimal('2')
    # quantize to 1 decimal place, always rounding down
    d_q = d.quantize(Decimal('0.1'), rounding=ROUND_DOWN)
    return float(d_q)

def compute_sell(row):
    buy = row['element_in_cost']
    now = row['now_cost']
    if now < buy:
        return now
    if now == buy:
        return buy
    profit = now - buy
    half = half_profit_round_down_to_0_1(profit)
    return buy + half

def get_squad_worth(entry_id, df, my_squad_15_array):
	transfers = requests.get(f"{BASE}/entry/{entry_id}/transfers/").json()
	transfers = pd.DataFrame(transfers)
	transfers = transfers.sort_values('time', ascending = False)
	transfers = transfers.drop_duplicates(subset=['element_in'], keep='first') # remove dupe in case player added more than once
	transfers = transfers[['element_in', 'element_in_cost']].merge(df[['player_id', 'player_name', 'now_cost']], left_on='element_in', right_on='player_id', how = 'left').drop_duplicates()
	transfers = transfers[transfers['player_name'].isin(my_squad_15_array)]
	transfers['element_in_cost'] = transfers['element_in_cost'] / 10
	transfers['sell_price'] = 0 
	transfers['sell_price'] = transfers.apply(compute_sell, axis=1)
	transfers['sell_price'] = np.round(transfers['sell_price'], 1)
	return transfers


def set_player_out(df: pd.DataFrame, player: str, *,
                        mean: float = 0.0,
                        median: float = 0.0,
                        prob_return: float = 0.0,
                        prob_goal: float = 0.0,
                        prob_double_digit: float = 0.0,
                        ci_lower: float = 0.0,
                        ci_upper: float = 0.0,
                        ci_95: tuple = (np.float64(0.0), np.float64(0.0)),
                        inplace: bool = True) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    if 'player' not in df.columns:
        raise KeyError("DataFrame must contain a 'player' column")

    print(f"Replacing data for {player}")
    mask = df['player'] == player
    if not mask.any():
        return df

    # Ensure scalar columns exist
    cols_defaults = {
        'mean': mean,
        'median': median,
        'prob_return': prob_return,
        'prob_goal': prob_goal,
        'prob_double_digit': prob_double_digit,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }
    for col, default in cols_defaults.items():
        if col not in df.columns:
            df[col] = np.nan
        df.loc[mask, col] = default

    #replacee the tuple in confidence_interval_95
    idx = df.index[df['player'] == player][0]
    df.at[idx, 'ci_95'] = ci_95
    return df

def transform_squad(squad_dict):
    # Flatten into array
    squad_array = []
    player_position_map = {}
    
    for position, players in squad_dict.items():
        for player in players:
            if player:  # skip empty entries
                squad_array.append(player)
                player_position_map[player] = position
    
    return squad_array, player_position_map


def set_player_out_multiple(predictions_list, player, weeks=None, delay = 0, **kwargs):
    """
    Apply set_player_out to multiple prediction DataFrames.

    Parameters
    ----------
    predictions_list : list of pd.DataFrame
        The list of prediction outputs (e.g. predictions_dfs).
    player : str
        The player name to update.
    weeks : int or None
        Number of weeks to apply the update. If None, applies to all.
    kwargs : dict
        Extra keyword arguments passed to set_player_out (mean, median, etc.).

    Returns
    -------
    updated_list : list of pd.DataFrame
        The updated DataFrames.
    """
    if weeks is None:
        weeks = len(predictions_list)  - delay

    updated_list = []
    for i, df in enumerate(predictions_list):
        if i < delay:
            # skip until delay is passed
            updated_list.append(df)
        elif i < delay + weeks:
            updated_df = set_player_out(df, player, **kwargs)
            updated_list.append(updated_df)
        else:
            updated_list.append(df)

    return updated_list