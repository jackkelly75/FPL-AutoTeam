import pandas as pd
import numpy as np

#Calaculte team stats and fixture difficulaty
import numpy as np
import pandas as pd
from math import exp, log
from scipy.optimize import minimize
from scipy.special import gammaln

def _log_poisson_pmf(k, lam):
    lam = np.maximum(lam, 1e-12)
    return k * np.log(lam) - lam - gammaln(k + 1)


def _match_result_probs(lambda_home, lambda_away, max_goals=6, rho=0.0):
    """
    Compute match result probabilities from independent Poissons then apply
    Dixon-Coles adjustment for low-score dependence using rho.
    Returns dict with keys 'home_win','draw','away_win','matrix'.
    """
    probs = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    # independent joint Poisson
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            probs[i, j] = np.exp(_log_poisson_pmf(i, lambda_home) + _log_poisson_pmf(j, lambda_away))

    # apply Dixon-Coles tau adjustment only for the four low-score cells
    # canonical mapping:
    # (0,0) -> 1 - rho
    # (1,1) -> 1 - rho
    # (0,1) -> 1 + rho
    # (1,0) -> 1 + rho
    if rho != 0.0:
        # ensure tau values are non-negative
        tau00 = max(1.0 - rho, 1e-12)
        tau11 = max(1.0 - rho, 1e-12)
        tau01 = max(1.0 + rho, 1e-12)
        tau10 = max(1.0 + rho, 1e-12)

        # only apply if those cells are within the computed grid
        if max_goals >= 0:
            probs[0, 0] *= tau00
        if max_goals >= 1:
            probs[1, 1] *= tau11
            probs[0, 1] *= tau01
            probs[1, 0] *= tau10

        # renormalize so the joint pmf sums to 1
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            raise ValueError("Invalid joint probability mass after Dixon-Coles adjustment.")
        probs /= total

    # aggregate to match outcomes
    away_win = probs[np.triu_indices(max_goals + 1, k=1)].sum()
    draw = np.sum(np.diag(probs))
    home_win = probs[np.tril_indices(max_goals + 1, k=-1)].sum()

    return {'home_win': home_win, 'draw': draw, 'away_win': away_win, 'matrix': probs}


def calculate_team_stats_dixon_coles_ewma(
    df,
    use_rho=True,
    rho_init=0.0,
    halflife_gw=8.0,
    maxiter=2000,
    tol=1e-9,
    clip_linpred=20.0
):
    """
    Safe Poisson MLE with Dixon-Coles and EWMA weights. Clips linear predictors to avoid overflow.
    Parameters same as before; clip_linpred caps the linear predictor (a + d + log_home) to +/- clip_linpred.
    """
    required = {'gameweek','team_name','opponent_team_name','was_home','G','goals_conceded'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Aggregate to team-game level and build fixtures (home rows)
    team_game = df.groupby(['gameweek','team_name','was_home','opponent_team_name'], as_index=False).agg({
        'G': 'sum',                     # goals scored by team: sum across players
        'goals_conceded': 'first'       # conceded is same for all players in that match
    })
    home_rows = team_game[team_game['was_home'] == True].copy()
    if home_rows.empty:
        raise ValueError("No home fixtures found. Check 'was_home' and 'opponent_team_name' columns.")

    fixtures = []
    for _, r in home_rows.iterrows():
        fixtures.append({
            'gameweek': int(r['gameweek']),
            'home': r['team_name'],
            'away': r['opponent_team_name'],
            'home_goals': int(r['G']),
            'away_goals': int(r['goals_conceded'])
        })

    teams = sorted(list({f['home'] for f in fixtures} | {f['away'] for f in fixtures}))
    n = len(teams)
    team_idx = {t:i for i,t in enumerate(teams)}

    GWs = np.array([f['gameweek'] for f in fixtures], dtype=int)
    H = np.array([team_idx[f['home']] for f in fixtures], dtype=int)
    A = np.array([team_idx[f['away']] for f in fixtures], dtype=int)
    GH = np.array([f['home_goals'] for f in fixtures], dtype=int)
    GA = np.array([f['away_goals'] for f in fixtures], dtype=int)

    current_gw = int(df['gameweek'].max())
    if halflife_gw <= 0:
        raise ValueError("halflife_gw must be > 0")
    age = current_gw - GWs
    weights = 0.5 ** (age / halflife_gw)
    weights = weights * (len(weights) / weights.sum())

    # Negative weighted log-likelihood with clipping and safety
    def neg_loglik(x):
        a = x[:n]
        d = x[n:2*n]
        log_home = x[2*n]
        rho = x[2*n+1] if use_rho else 0.0

        # vectorized linear predictors (clipped)
        lin_h = a[H] + d[A] + log_home
        lin_a = a[A] + d[H]
        # clip to avoid overflow in exp
        lin_h = np.clip(lin_h, -clip_linpred, clip_linpred)
        lin_a = np.clip(lin_a, -clip_linpred, clip_linpred)
        lam_h = np.exp(lin_h)
        lam_a = np.exp(lin_a)

        # compute weighted log-likelihood vectorized
        ll_vec = weights * (_log_poisson_pmf(GH, lam_h) + _log_poisson_pmf(GA, lam_a))

        # Dixon-Coles adjustments for low scores (vectorized)
        if use_rho:
            # mask where both scores <=1
            mask = (GH <= 1) & (GA <= 1)
            if mask.any():
                g = GH[mask]; ga = GA[mask]
                # compute tau per pair
                tau = np.ones_like(g, dtype=float)
                # (0,0) and (1,1) -> 1 - rho ; (0,1) and (1,0) -> 1 + rho
                tau[(g==0) & (ga==0)] = 1.0 - rho
                tau[(g==1) & (ga==1)] = 1.0 - rho
                tau[(g==0) & (ga==1)] = 1.0 + rho
                tau[(g==1) & (ga==0)] = 1.0 + rho
                tau = np.maximum(tau, 1e-8)
                ll_vec[mask] += weights[mask] * np.log(tau)

        ll = np.sum(ll_vec)

        # if ll is nan or inf, return large penalty
        if not np.isfinite(ll):
            return 1e12 + np.sum(np.abs(x))  # penalize large params
        return -ll

    # initial guess and constraints/bounds
    x0 = np.concatenate([np.zeros(n), np.zeros(n), np.array([0.1]), np.array([rho_init if use_rho else 0.0])])
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x[:n])})
    bnds = [(None, None)] * (2*n + 1) + [(-0.99, 0.99)]

    # Use SLSQP; if it still struggles try 'trust-constr' or L-BFGS-B with reparameterization
    res = minimize(neg_loglik, x0, method='SLSQP', constraints=cons, bounds=bnds,
                   options={'maxiter': maxiter, 'ftol': tol})

    if not res.success:
        raise RuntimeError("MLE failed: " + res.message)

    x_hat = res.x
    a_hat = x_hat[:n]
    d_hat = x_hat[n:2*n]
    log_home_hat = x_hat[2*n]
    rho_hat = x_hat[2*n+1] if use_rho else 0.0

    attack = np.exp(a_hat)
    defence = np.exp(-d_hat)
    out = pd.DataFrame({'team': teams, 'attack': attack, 'defence': defence}).set_index('team')
    out['attack_home'] = out['attack'] * exp(log_home_hat)
    out['attack_away'] = out['attack']
    out['defence_home'] = out['defence']
    out['defence_away'] = out['defence']
    out['rho'] = rho_hat

    def predict_fixture(home_team, away_team, max_goals=6):
        if home_team not in out.index or away_team not in out.index:
            raise ValueError("Teams not found in ratings.")
        a_map = {teams[i]: a_hat[i] for i in range(n)}
        d_map = {teams[i]: d_hat[i] for i in range(n)}
        lin_h = a_map[home_team] + d_map[away_team] + log_home_hat
        lin_a = a_map[away_team] + d_map[home_team]
        lin_h = np.clip(lin_h, -clip_linpred, clip_linpred)
        lin_a = np.clip(lin_a, -clip_linpred, clip_linpred)
        lam_h = exp(lin_h); lam_a = exp(lin_a)
        probs = _match_result_probs(lam_h, lam_a, max_goals=max_goals)
        return {'expected_goals_home': lam_h, 'expected_goals_away': lam_a, 'probabilities': probs, 'rho': rho_hat}

    meta = {}
    meta['predict_fixture'] = predict_fixture
    meta['a_hat'] = a_hat
    meta['d_hat'] = d_hat
    meta['log_home'] = log_home_hat
    meta['rho'] = rho_hat
    meta['weights'] = weights
    meta['halflife_gw'] = halflife_gw

    return out, meta


def calculate_team_stats(df):
    """
    Calculate offensive and defensive strength for each team based on 
    goals scored and conceded across all games played so far.
    """
    team_stats = {}
    
    # Get unique teams
    teams = pd.concat([df['team_name'], df['opponent_team_name']]).unique()
    
    for team in teams:
        # Goals scored by this team (rows where team_name is this team)
        goals_for = df[df['team_name'] == team]['G'].sum()
        games_for = len(df[df['team_name'] == team]['gameweek'].unique())
        
        # Goals conceded by this team (rows where opponent_team_name is this team)
        goals_against = df[df['opponent_team_name'] == team]['G'].sum()
        games_against = len(df[df['opponent_team_name'] == team]['gameweek'].unique())
        
        # Calculate averages per game
        avg_goals_for = goals_for / games_for if games_for > 0 else 0
        avg_goals_against = goals_against / games_against if games_against > 0 else 0
        
        team_stats[team] = {
            'goals_for': goals_for,
            'goals_against': goals_against,
            'games_played': games_for,
            'avg_goals_for': avg_goals_for,
            'avg_goals_against': avg_goals_against,
            'offensive_strength': avg_goals_for,  # Higher = better attack
            'defensive_strength': 1 / (avg_goals_against + 0.1)  # Higher = better defence (avoid division by 0)
        }
    
    return pd.DataFrame(team_stats).T.sort_values('offensive_strength', ascending=False)

def calculate_fixture_difficulty(fixtures, team_stats, normalize=True):
    """
    Calculate difficulty rating (0-10) for each fixture based on opponent's defensive strength.
    
    Parameters:
    - fixtures: list of tuples [(home_team, away_team), ...]
    - team_stats: dataframe from calculate_team_stats()
    - normalize: if True, scales to 0-10; if False, returns raw difficulty scores
    """
    fixture_difficulty = []
    
    for home_team, away_team in fixtures:
        if home_team not in team_stats.index or away_team not in team_stats.index:
            print(f"Warning: {home_team} or {away_team} not found in team stats")
            continue
        
        # For home team: difficulty is based on away team's defensive strength
        home_difficulty = team_stats.loc[away_team, 'defensive_strength']
        
        # For away team: difficulty is based on home team's defensive strength
        away_difficulty = team_stats.loc[home_team, 'defensive_strength']
        
        fixture_difficulty.append({
            'home_team': home_team,
            'away_team': away_team,
            'matchup': f"{home_team} vs {away_team}",
            'home_difficulty_raw': home_difficulty,
            'away_difficulty_raw': away_difficulty
        })
    
    difficulty_df = pd.DataFrame(fixture_difficulty)
    
    if normalize:
        # Normalize to 0-10 scale (lower is easier, higher is harder)
        # Inverse the difficulty score so higher defensive strength = harder to score against
        min_diff = difficulty_df[['home_difficulty_raw', 'away_difficulty_raw']].min().min()
        max_diff = difficulty_df[['home_difficulty_raw', 'away_difficulty_raw']].max().max()
        
        if max_diff > min_diff:
            difficulty_df['home_difficulty'] = 10 * (difficulty_df['home_difficulty_raw'] - min_diff) / (max_diff - min_diff)
            difficulty_df['away_difficulty'] = 10 * (difficulty_df['away_difficulty_raw'] - min_diff) / (max_diff - min_diff)
        else:
            difficulty_df['home_difficulty'] = 5
            difficulty_df['away_difficulty'] = 5
    else:
        difficulty_df['home_difficulty'] = difficulty_df['home_difficulty_raw']
        difficulty_df['away_difficulty'] = difficulty_df['away_difficulty_raw']
    
    return difficulty_df[['matchup', 'home_team', 'away_team', 'home_difficulty', 'away_difficulty']].round(2)

