import pandas as pd
import numpy as np
from scipy.stats import poisson
# Monte coarlo
def get_player_historical_stats(df, player_name, element_type=None):
    """
    Get historical performance stats for a player.
    Returns metrics that are relevant for scoring FPL points.
    """
    player_data = df[df['player_name'] == player_name]
    
    if player_data.empty:
        return None
    
    # Filter by position if provided
    if element_type:
        player_data = player_data[player_data['element_type'] == element_type]
    
    if player_data.empty:
        return None
    
    # Key scoring metrics
    stats_dict = {
        'player_name': player_name,
        'position': player_data['element_type'].iloc[0],
        'team': player_data['team_name'].iloc[0],
        'games_played': len(player_data),
        'avg_points': player_data['total_points'].mean(),
        'std_points': player_data['total_points'].std(),
        'min_points': player_data['total_points'].min(),
        'max_points': player_data['total_points'].max(),
        'avg_minutes': player_data['minutes'].mean(),
        'minutes_played_rate': (player_data['minutes'] > 0).mean(),  # % of games played
        'avg_xg': player_data['xG'].astype(float).mean(),
        'avg_xa': player_data['xassists'].astype(float).mean(),
        'avg_g': player_data['G'].astype(float).mean(),
        'avg_a': player_data['assists'].astype(float).mean(),
        'avg_goals_conceded': player_data['goals_conceded'].astype(float).mean(),
        'avg_x_goals_conceded': player_data['x_goals_conceded'].astype(float).mean(),
        'avg_defensive_contribution': player_data['defensive_contribution'].astype(float).mean(),
        'avg_saves': player_data['saves'].astype(float).mean(),
        #'avg_shots': player_data['shots'].mean(),
        #'avg_sot': player_data['SoT'].mean(),
        #'avg_key_passes': player_data['key_passes'].mean(),
        #'avg_tackles': player_data['tackles'].mean(),
        #'avg_cs': player_data['CS'].mean() if 'CS' in player_data.columns else 0,
        'all_games': player_data
    }
    
    return stats_dict


def compute_player_rates(player_stats_df):
    """
    Compute xG, xA, shots, mins etc from historical per-90 values.
    """

    total_minutes = player_stats_df['minutes'].sum()
    if total_minutes == 0:
        return None
    
    rate = lambda col: player_stats_df[col].sum() / (total_minutes / 90)

    return {
        'xG_per90': rate('xG'),
        'xA_per90': rate('xassists'),
        'saves_per90': rate('saves'),
        'gc_per90': rate('goals_conceded'),
        'yellow_rate': player_stats_df['yellow_cards'].mean(),
        'red_rate': player_stats_df['red_cards'].mean(),
        'og_rate': player_stats_df['own_goals'].mean(),
        'minutes_start_prob': np.mean(player_stats_df['minutes'] >= 60),
        'minutes_cameo_prob': np.mean((player_stats_df['minutes'] > 0) & (player_stats_df['minutes'] < 60)),
        'minutes_dnp_prob': np.mean(player_stats_df['minutes'] == 0)
    }
def allocate_team_xg_to_player(player_rates, team_xg, team_avg_xG_per90):
    """
    Distribute team xG according to player's historical share of xG.
    """

    if team_avg_xG_per90 == 0:
        return 0.0

    share = player_rates['xG_per90'] / team_avg_xG_per90
    return team_xg * share

def simulate_minutes(player_rates):
    r = np.random.rand()

    if r < player_rates['minutes_dnp_prob']:
        return 0
    elif r < player_rates['minutes_dnp_prob'] + player_rates['minutes_cameo_prob']:
        return np.random.randint(1, 30)
    else:
        return int(np.random.normal(80, 8))  # start



def fpl_points_from_events(position, goals, assists, cs, gc, saves, minutes, yc, rc, og):
    
    points = 0

    # Minutes
    if minutes >= 60:
        points += 2
    elif minutes > 0:
        points += 1

    # Goals
    if position == "FWD":
        points += goals * 4
    elif position == "MID":
        points += goals * 5
    elif position in ("DEF", "GK"):
        points += goals * 6

    # Assists
    points += assists * 3

    # Clean sheet
    if cs:
        if position == "DEF": points += 4
        if position == "GK": points += 4
        if position == "MID": points += 1

    # Goals conceded
    if position in ("GK", "DEF"):
        points -= (gc // 2)

    # Saves
    if position == "GK":
        points += (saves // 3)

    # Cards
    points -= yc * 1
    points -= rc * 3

    # Own goals
    points -= og * 2

    return points



def monte_carlo_player_prediction(
    player_name,
    player_position,                        # "GK", "DEF", "MID", "FWD"
    player_stats_df,
    team_xg,                                 # derived from Dixon–Coles (e.g. home_pred_score)
    opponent_xg,                             # opponent expected goals
    team_avg_xG_per90,
    n_simulations=20000
):
    
    player_rates = compute_player_rates(player_stats_df)
    if player_rates is None:
        return None

    # Expected involvement
    player_exp_xG = allocate_team_xg_to_player(player_rates, team_xg, team_avg_xG_per90)
    player_exp_xA = player_rates['xA_per90'] * (team_xg / team_avg_xG_per90)

    results = []

    for _ in range(n_simulations):

        # Minutes model
        minutes = simulate_minutes(player_rates)
        if minutes == 0:
            results.append(0)
            continue

        # Scale event rates by minutes
        scale = minutes / 90

        # Goals & assists
        goals = np.random.poisson(player_exp_xG * scale)
        assists = np.random.poisson(player_exp_xA * scale)

        # Opponent attacking output → goals conceded
        gc = np.random.poisson(opponent_xg)

        # Clean sheet
        cs = int(gc == 0)

        # GK saves
        saves = 0
        if player_position == "GK":
            saves = np.random.poisson(player_rates['saves_per90'] * scale)

        # Cards
        yc = np.random.binomial(1, player_rates['yellow_rate'])
        rc = np.random.binomial(1, player_rates['red_rate'])

        # Own goals
        og = np.random.binomial(1, player_rates['og_rate'])

        # Convert to points
        pts = fpl_points_from_events(
            player_position,
            goals, assists, cs, gc, saves, minutes, yc, rc, og
        )

        results.append(pts)

    results = np.array(results)

    return {
        "player": player_name,
        "mean": round(results.mean(), 2),
        "median": round(np.median(results), 2),
        "ci_95": (round(np.percentile(results, 2.5), 2),
                  round(np.percentile(results, 97.5), 2)),
        "prob_return": round(np.mean(results >= 6), 3),
        "prob_goal": round(np.mean(results >= 4), 3),
        "prob_double_digit": round(np.mean(results >= 10), 3),
        "distribution": results
    }



# Predict fixutres
def predict_fixture(df, fixtures, team_stats, predict_fixture_func,  n_simulations=10000):
    """
    Predict all players for all fixtures in fixtures.
    
    Parameters:
    - df: historical player data
    - fixtures: list of tuples [(home_team, away_team), ...]
    - team_stats: dataframe with defensive and offensive strength
    - n_simulations: number of MC simulations per player
    """
    
    all_predictions = []
    
    for home_team, away_team in fixtures:
        print(f"\nProcessing: {home_team} vs {away_team}")
        # Get team stats
        if home_team not in team_stats.index or away_team not in team_stats.index:
            print(f"  Warning: Team stats not found for {home_team} or {away_team}")
            continue
        
        pred = predict_fixture_func(home_team, away_team)
        home_pred_score = pred['expected_goals_home']
        away_pred_score = pred['expected_goals_away']

        home_def_strength = team_stats.loc[home_team, 'defence_home']
        home_off_strength = team_stats.loc[home_team, 'attack_home']
        away_def_strength = team_stats.loc[away_team, 'defence_away']
        away_off_strength = team_stats.loc[away_team, 'attack_away']
        
        # Normalize defensive strength to 0-10 scale if needed
        # (convert from raw defensive_strength metric to difficulty)
        max_def = max([team_stats['defence_home'].max(), team_stats['defence_away'].max()])
        min_def = min([team_stats['defence_home'].min(), team_stats['defence_away'].min()])
        max_off = max([team_stats['attack_home'].max(), team_stats['attack_away'].max()])
        min_off = min([team_stats['attack_home'].min(), team_stats['attack_away'].min()])
        
        home_def_normalized = 10 * (home_def_strength - min_def) / (max_def - min_def) if max_def > min_def else 5
        away_def_normalized = 10 * (away_def_strength - min_def) / (max_def - min_def) if max_def > min_def else 5
        home_off_normalized = 10 * (home_off_strength - min_off) / (max_off - min_off)
        away_off_normalized = 10 * (away_off_strength - min_off) / (max_off - min_off)
        
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

        #calcaulte team xG per match
        team_df = df[df['team_name'] == home_team]
        home_xg_per_match = team_df.groupby('gameweek')['xG'].sum().mean()
        team_df = df[df['team_name'] == away_team]
        away_xg_per_match = team_df.groupby('gameweek')['xG'].sum().mean()

        # Predict home team players
        home_players = df[df['team_name'] == home_team]['player_name'].unique()
        for player in home_players:
            player_stats = get_player_historical_stats(df, player)
            if player_stats is None:
                continue

            player_stats['position'] = position_map.get(int(player_stats['position']), 'UNK')
            pred = monte_carlo_player_prediction(
                player_name=player,
                player_position=player_stats['position'],
                player_stats_df=player_stats['all_games'],   # dataframe of this player's historic games
                team_xg=home_pred_score,        # from Dixon–Coles
                opponent_xg=away_pred_score,
                team_avg_xG_per90=home_xg_per_match           # from season team stats
            )            
            pred['matchup'] = f"{home_team} vs {away_team}"
            pred['team'] = home_team
            pred['opponent'] = away_team
            pred['venue'] = 'Home'
            pred['position'] = player_stats['position']
            pred['fixture_difficulty'] = away_def_normalized
            all_predictions.append(pred)

        # Predict away team players
        away_players = df[df['team_name'] == away_team]['player_name'].unique()
        for player in away_players:
            player_stats = get_player_historical_stats(df, player)
            if player_stats is None:
                continue
            player_stats['position'] = position_map.get(int(player_stats['position']), 'UNK')

            pred = monte_carlo_player_prediction(
                player_name=player,
                player_position=player_stats['position'],
                player_stats_df=player_stats['all_games'],   # dataframe of this player's historic games
                team_xg=away_pred_score,        # from Dixon–Coles
                opponent_xg=home_pred_score,
                team_avg_xG_per90=away_xg_per_match           # from season team stats
            )                        
            pred['matchup'] = f"{home_team} vs {away_team}"
            pred['team'] = away_team
            pred['opponent'] = home_team
            pred['venue'] = 'Away'
            pred['position'] = player_stats['position']
            pred['fixture_difficulty'] = home_def_normalized
            all_predictions.append(pred)
    
    all_predictions = pd.DataFrame(all_predictions)
    all_predictions[["ci_lower", "ci_upper"]] = all_predictions["ci_95"].apply(get_bounds)
    return all_predictions

def get_bounds(val):
    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return pd.Series([val[0], val[1]])
    if isinstance(val, np.ndarray) and val.size >= 2:
        return pd.Series([val[0], val[1]])
    if isinstance(val, (float, int, np.floating)):
        return pd.Series([float(val), np.nan])
    if isinstance(val, str):
        try:
            s = val.strip().strip("()")
            parts = [p.strip() for p in s.split(",")]
            lower = float(parts[0]) if parts and parts[0] != "" else np.nan
            upper = float(parts[1]) if len(parts) > 1 and parts[1] != "" else np.nan
            return pd.Series([lower, upper])
        except Exception:
            return pd.Series([np.nan, np.nan])
    return pd.Series([np.nan, np.nan])


# Find starting line up
def get_position_from_id(position_id):
    """
    Convert element_type ID to position string.
    1=GK, 2=DEF, 3=MID, 4=FWD
    """
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    return position_map.get(position_id, 'Unknown')

def get_squad_players_positions(my_squad_15, df):
    """
    Get position information for each player in squad.
    Returns dict with player_name: position
    """
    squad_positions = {}
    
    for position, players in my_squad_15.items():
        for player in players:
            # Verify player exists in data and get their position ID
            player_data = df[df['player_name'] == player]
            if not player_data.empty:
                position_id = player_data['element_type'].iloc[0]
                squad_positions[player] = get_position_from_id(position_id)
            else:
                # Fallback to provided position if not in data
                squad_positions[player] = position
    
    return squad_positions

def optimize_lineups_for_weeks(predictions_dfs, my_squad_15, df, 
                               constraints=None):
    """
    Optimize starting 11 for each gameweek using provided constraints.
    
    Parameters:
    - predictions_dfs: dict, list, or single dataframe
        dict: {1: predictions_df_1, 2: predictions_df_2, ...}
        list: [predictions_df_1, predictions_df_2, ...]
        dataframe: predictions for a single week
    - my_squad_15: dict with squad structure
    - df: historical dataframe (to get position info)
    - constraints: dict with lineup constraints
        Default: {'GK': 1, 'DEF': {'min': 3, 'max': 5}, 
                  'MID': {'min': 2, 'max': 5}, 'FWD': {'min': 1, 'max': 3}}
    
    Returns:
    - dict with optimized lineups for each week
    """
    
    if constraints is None:
        constraints = {
            'GK': 1,
            'DEF': {'min': 3, 'max': 5},
            'MID': {'min': 2, 'max': 5},
            'FWD': {'min': 1, 'max': 3}
        }
    
    # Get squad structure
    squad_positions = get_squad_players_positions(my_squad_15, df)
    
    # Flatten squad into list
    all_squad_players = []
    for position, players in my_squad_15.items():
        for player in players:
            all_squad_players.append({
                'name': player,
                'position': squad_positions[player]
            })
    
    squad_df = pd.DataFrame(all_squad_players)
    
    # Handle dict, list, or single dataframe
    if isinstance(predictions_dfs, dict):
        pred_dict = predictions_dfs
    elif isinstance(predictions_dfs, list):
        pred_dict = {i+1: pred_df for i, pred_df in enumerate(predictions_dfs)}
    elif isinstance(predictions_dfs, pd.DataFrame):
        pred_dict = {1: predictions_dfs}   # single week
    else:
        raise ValueError("predictions_dfs must be dict, list, or dataframe")
    
    optimized_lineups = {}
    
    for week, predictions_df in pred_dict.items():
        print(f"\n{'='*80}")
        print(f"OPTIMIZING LINEUPS FOR GAMEWEEK {week}")
        print(f"{'='*80}")
        
        # Filter predictions to only squad players
        squad_predictions = predictions_df[
            predictions_df['player'].isin([p['name'] for p in all_squad_players])
        ].copy()
        
        if squad_predictions.empty:
            print(f"Warning: No squad players found in predictions for week {week}")
            continue
        
        # Add position info to predictions
        squad_predictions['position'] = squad_predictions['player'].map(
            lambda x: next((p['position'] for p in all_squad_players if p['name'] == x), None)
        )
        
        # Find optimal lineup
        best_lineup, best_score = find_best_lineup(
            squad_predictions,
            constraints,
            squad_df
        )
        
        optimized_lineups[week] = {
            'lineup': best_lineup,
            'total_points': best_score,
            'predictions_df': squad_predictions
        }
        
        # Display results
        print_lineup_results(week, best_lineup, best_score, squad_predictions)
    
    return optimized_lineups

def find_best_lineup(squad_predictions, constraints, squad_df):
    """
    Find the optimal 11-player lineup that maximizes expected points
    while respecting position constraints.
    
    Uses intelligent search to avoid checking all combinations (too slow).
    """
    
    best_lineup = None
    best_score = -np.inf
    
    # Group players by position
    by_position = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_players = squad_predictions[squad_predictions['position'] == pos].copy()
        pos_players = pos_players.sort_values('mean', ascending=False)
        by_position[pos] = pos_players
    
    # Generate all valid combinations respecting constraints
    gk_constraint = constraints['GK']
    def_constraint = constraints['DEF']
    mid_constraint = constraints['MID']
    fwd_constraint = constraints['FWD']
    
    # Iterate through valid combinations
    for n_def in range(def_constraint['min'], def_constraint['max'] + 1):
        for n_mid in range(mid_constraint['min'], mid_constraint['max'] + 1):
            for n_fwd in range(fwd_constraint['min'], fwd_constraint['max'] + 1):
                # Check if total is 11 (1 GK + DEF + MID + FWD)
                if 1 + n_def + n_mid + n_fwd != 11:
                    continue
                
                # For each valid formation, pick top scorers at each position
                best_at_pos = {
                    'GK': by_position['GK'].head(gk_constraint),
                    'DEF': by_position['DEF'].head(n_def),
                    'MID': by_position['MID'].head(n_mid),
                    'FWD': by_position['FWD'].head(n_fwd)
                }
                
                # Combine and calculate score
                lineup = pd.concat(best_at_pos.values())
                score = lineup['mean'].sum()
                
                if score > best_score:
                    best_score = score
                    best_lineup = lineup
    
    return best_lineup, best_score

def print_lineup_results(week, best_lineup, best_score, squad_predictions):
    """
    Print formatted lineup results for a gameweek.
    """
    print(f"\nOptimal Starting XI:")
    print(f"Expected Total Points: {best_score:.2f}\n")
    
    # Print by position
    positions_order = ['GK', 'DEF', 'MID', 'FWD']
    
    for pos in positions_order:
        pos_players = best_lineup[best_lineup['position'] == pos].sort_values('mean', ascending=False)
        
        if len(pos_players) > 0:
            print(f"{pos} ({len(pos_players)} players):")
            for _, player in pos_players.iterrows():
                print(f"  {player['player']:25} vs {player['opponent']:15} " +
                      f"({player['venue']:4}) - {player['mean']:6.2f} pts " +
                      f"(CI: {player['ci_lower']:5.1f}-{player['ci_upper']:5.1f}) " +
                      f"Prob 10+: {player['prob_double_digit']:.2f}")
            print()
    
    # Show who's benched
    bench_players = squad_predictions[~squad_predictions['player'].isin(best_lineup['player'])]
    print(f"Bench ({len(bench_players)} players):")
    bench_sorted = bench_players.sort_values('mean', ascending=False)
    for _, player in bench_sorted.iterrows():
        print(f"  {player['player']:25} vs {player['opponent']:15} " +
              f"({player['venue']:4}) - {player['mean']:6.2f} pts")

def print_summary_comparison(optimized_lineups):
    """
    Print summary comparison across all weeks.
    """
    print(f"\n{'='*80}")
    print("GAMEWEEK SUMMARY")
    print(f"{'='*80}\n")
    
    summary_data = []
    for week in sorted(optimized_lineups.keys()):
        summary_data.append({
            'Week': week,
            'Expected Points': optimized_lineups[week]['total_points'],
            'Lineup Size': len(optimized_lineups[week]['lineup'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(f"\nTotal Expected Points (7 weeks): {summary_df['Expected Points'].sum():.2f}")
    print(f"Average per week: {summary_df['Expected Points'].mean():.2f}")

def get_transfer_suggestions(optimized_lineups, my_squad_15, predictions_dfs):
    """
    Suggest which players to transfer based on predicted performance across weeks.
    """
    print(f"\n{'='*80}")
    print("TRANSFER SUGGESTIONS")
    print(f"{'='*80}\n")
    
    # Flatten all predictions
    all_preds = []
    for week, pred_df in (predictions_dfs.items() if isinstance(predictions_dfs, dict) else enumerate(predictions_dfs, 1)):
        pred_df_copy = pred_df.copy()
        pred_df_copy['week'] = week
        all_preds.append(pred_df_copy)
    
    combined_preds = pd.concat(all_preds, ignore_index=True)
    
    # Get squad players
    all_squad_players = []
    for position, players in my_squad_15.items():
        all_squad_players.extend(players)
    
    # Calculate 7-week average for each squad player
    player_7week = combined_preds[combined_preds['player_name'].isin(all_squad_players)].groupby('player_name').agg({
        'predicted_points_mean': ['mean', 'sum', 'std'],
        'prob_double_digit': 'mean'
    }).round(2)
    
    player_7week.columns = ['avg_per_week', 'total_7weeks', 'std_dev', 'avg_prob_double_digit']
    player_7week = player_7week.sort_values('total_7weeks', ascending=False)
    
    print("Squad Performance Over 7 Weeks:")
    print(player_7week.to_string())
    
    print(f"\n\nPlayers underperforming (avg < 4 pts/week):")
    underperformers = player_7week[player_7week['avg_per_week'] < 4]
    if len(underperformers) > 0:
        print(underperformers.to_string())
        print("\nConsider transferring these players out")
    else:
        print("No major underperformers")
