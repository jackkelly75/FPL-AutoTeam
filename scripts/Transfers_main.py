# Functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import heapq
from typing import Dict, List, Tuple, Any
from decimal import Decimal, ROUND_DOWN
import sys
#import custom functions
from fpl_predictor import transfer_recom
from fpl_predictor import team_prob
from fpl_predictor import fixture_transfer
from fpl_predictor import import_data_func


#Set Squad info - does not import from FPL as this will not be most recent team
entry_id = 4311890   # replace with your entry id
left_in_budget = 2.6
gw = 17

my_squad_15 = {
        'GK': ['Petrović', 'Sánchez'],
        'DEF': ['J.Timber', 'Kayode',  'Van den Berg', 'Chalobah', 'Tarkowski'],
        'MID': ['Saka', 'Anthony', 'Enzo', 'Wilson', 'Kudus'],
        'FWD': ['Woltemade', 'Thiago', 'Haaland'],
    }

my_squad_15_array, player_position_map = import_data_func.transform_squad(my_squad_15)


# Import data
#Import stats for all players who have played at least one minute this season
df = import_data_func.import_player_data(num_mins = 1)
'''
['player_id', 'first_name', 'second_name', 'team_id', 'team_name',
       'element_type', 'gameweek', 'kickoff_time', 'minutes', 'G', 'xG',
       'assists', 'xassists', 'clean_sheets', 'goals_conceded',
       'x_goals_conceded', 'own_goals', 'clearances_blocks_interceptions',
       'defensive_contribution', 'penalties_saved', 'penalties_missed',
       'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps', 'influence',
       'creativity', 'threat', 'ict_index', 'total_points', 'was_home',
       'opponent_team_name', 'now_cost', 'player_name']
'''
#drop this bloke out
df = df[~((df.first_name == 'Callum') & (df.second_name == 'Wilson'))]


# ----- Cost of players
#cost of all players to buy now
player_cost = df[(df.gameweek == max(df.gameweek))][['player_name', 'now_cost']]
#cost of my players if sold
transfers = import_data_func.get_squad_worth(entry_id, df, my_squad_15_array)
squad_cost = round(transfers['sell_price'].sum(), 1)
squad_cost = squad_cost + left_in_budget

constraints = import_data_func.constraints


# Loop through fixtures up to week ahead you want
fixtures = []
for i in range(7): # next 7 gameweeks
    fixtures.append(import_data_func.fixtures_for_event(gw + i))


# ----------- Find predictiosn
#use dixon coles model to predict next gameweeks
team_stats, meta = team_prob.calculate_team_stats_dixon_coles_ewma(df, use_rho=True, rho_init=0.0, halflife_gw=5.0)


'''
pred = meta['predict_fixture']('Man Utd','Arsenal')
print(pred['expected_goals_home'], pred['expected_goals_away'])
print(pred['probabilities']['home_win'], pred['probabilities']['draw'], pred['probabilities']['away_win'])
'''

# Run predictions for all fixtures in the fixtures list
predictions_dfs = []

for fixture in fixtures:
    pred_df = fixture_transfer.predict_fixture(
        df,
        fixture,
        team_stats,
        predict_fixture_func=meta['predict_fixture'],
        n_simulations=10000
    )
    #pred_df[["ci_lower", "ci_upper"]] = pred_df["ci_95"].apply(fixture_transfer.get_bounds)
    predictions_dfs.append(pred_df)


#-----------------
# Handling injured or banned players
#-----------------
# I'm sure there is a way to import injured players from API
# Instea I have just supplied players I have on my team or big players who won't be playing
# This will impute thier predictions for these weeks with 0
predictions_dfs = import_data_func.set_player_out_multiple(predictions_dfs,  'Longstaff', weeks=5)
predictions_dfs = import_data_func.set_player_out_multiple(predictions_dfs,  'Muñoz', weeks=4)

#There is if AFCON is happening can bulk remove from list
afcon = ['Salah','Ouattara','Foster','Sarr','Ndiaye','Iwobi',
'Salah','Ait-Nouri','Marmoush','Mbeumo','Wissa','Aina',
'Wan-Bissaka','Diouf','Agbadou']

#if afcon is in future, you can set delay within set_player_out_multiple() for the number of weeks in future it is
for player_acon in afcon:
    predictions_dfs = import_data_func.set_player_out_multiple(predictions_dfs,  player_acon, weeks=4)






#---------
# View expected top 20 players for next gameweek
#---------
print("#------------expected top 20 players for next gameweek")
top_20 = predictions_dfs[0].nlargest(20, 'mean')
print(top_20[['player', 'position', 'team', 'opponent', 'venue', 'mean', 'ci_lower',
    'ci_upper', 'prob_double_digit']].to_string(index=False))
print("#------------highest upper confidence interval players for next gameweek")
upside = predictions_dfs[0].nlargest(10, 'ci_upper')
print(upside[['player', 'position', 'team', 'opponent', 'venue', 'mean', 'ci_lower',
	'ci_upper', 'prob_double_digit']].to_string(index=False))


#---------
# Select startting 11 from current team linup
#---------
optimized_lineups = fixture_transfer.optimize_lineups_for_weeks(predictions_dfs, my_squad_15, df, constraints)




# combine all weeks into one dataframe
predictions_combined = []
for i, preds in enumerate(predictions_dfs, start=1):
    tmp = preds.copy()
    tmp['week'] = i
    predictions_combined.append(tmp)

predictions_combined = pd.concat(predictions_combined, ignore_index=True)




# --- CONFIGURABLE PARAMETERS ---
HORIZON_WEEKS = 7            # number of weeks to optimize over
BUDGET_CAP = squad_cost
FREE_TRANSFERS_PER_WEEK = 1
MAX_BANKED_FREE = 5
PAID_TRANSFER_COST = 4       # points per extra transfer
BEAM_WIDTH = 50             # beam search width (increase for better results, slower)
MAX_PAID_TRANSFERS_TO_CONSIDER = 5  # per week when exploring (keeps branching manageable)
BENCH_WEIGHT = 0.1          # weight for bench predicted points in objective

# --- USER-SUPPLIED OBJECTS (placeholders for your environment) ---
# my_squad_15: dict with keys 'GK','DEF','MID','FWD' -> lists of player names (15 total)
# predictions_combined: DataFrame with columns ['player_name','week','predicted_points_mean']
# player_cost: DataFrame with columns ['player_name','now_cost']
# optimize_lineups_for_weeks(predictions_df, squad_15, df, constraints) -> returns dict-like with [1]['total_points']

player_position_map = predictions_combined.set_index('player')['position'].to_dict()
plan = transfer_recom.plan_transfers_beam_search(my_squad_15, predictions_combined, player_cost, player_position_map, fixture_transfer.optimize_lineups_for_weeks, afcon_bonus_week =1, horizon_weeks = HORIZON_WEEKS , beam_width = BEAM_WIDTH, budget_cap = BUDGET_CAP, df=df, constraints=constraints)

#edit 
# --- Notes and tuning ---
# - Increase BEAM_WIDTH for better search (slower).
# - Increase MAX_PAID_TRANSFERS_TO_CONSIDER if you want to explore more paid-transfer combos.
# - The candidate generation uses cumulative predicted points; you can refine it to use per-week matchups, fixture difficulty, or rotation risk.
# - If you have a reliable player->position mapping and a larger universe of players, consider filtering available candidates by team fixtures or minutes-played likelihood.
# - The bench proxy is a heuristic; if your optimize_lineups_for_weeks returns bench details, replace the bench proxy with exact bench points.


def compare_by_position(a, b):
    for role in sorted(set(a) | set(b)):
        list_a = a.get(role, [])
        list_b = b.get(role, [])
        # compare up to the longer list
        for i, (x, y) in enumerate(zip(list_a, list_b)):
            if x != y:
                print(f"{role}: {x} -> {y}")
        # handle extra items if lengths differ
        if len(list_a) > len(list_b):
            for extra in list_a[len(list_b):]:
                print(f"{role}: {extra} -> (missing)")
        elif len(list_b) > len(list_a):
            for extra in list_b[len(list_a):]:
                print(f"{role}: (missing) -> {extra}")

compare_by_position(my_squad_15, plan['history'][0]['squad'])

print("#-------- Starting 11--------#")
print(plan['history'][0]['optimized_lineup']['lineup'])
print("#-------- Captain--------#")
pd.set_option('display.max_columns', 4)
print(plan['history'][0]['optimized_lineup']['lineup'].sort_values('mean', ascending = False)[['player', 'mean', 'ci_95', 'prob_double_digit']].head(3))


compare_by_position(my_squad_15, plan['history'][0]['squad'])
print("----")
compare_by_position(plan['history'][0]['squad'], plan['history'][1]['squad'])
print("----")
compare_by_position(plan['history'][1]['squad'], plan['history'][2]['squad'])
print("----")
compare_by_position(plan['history'][2]['squad'], plan['history'][3]['squad'])
print("----")
compare_by_position(plan['history'][3]['squad'], plan['history'][4]['squad'])
print("----")
compare_by_position(plan['history'][4]['squad'], plan['history'][5]['squad'])
print("----")
compare_by_position(plan['history'][5]['squad'], plan['history'][6]['squad'])


my_squad_15 = {'GK': ['Petrović', 'Donnarumma'],
 'DEF': ['Senesi', 'Chalobah', 'Muñoz', 'Tarkowski', 'Van den Berg'],
 'MID': ['Saka', 'Enzo', 'J.Palhinha', 'Longstaff', 'Kudus'],
 'FWD': ['Woltemade', 'Thiago', 'Haaland']}

optimized_lineups = optimize_lineups_for_weeks(predictions_dict, my_squad_15, df, constraints)
