# ============================================================================
# COLLE CE BLOC DANS nba_analyzer_improved.py
# Juste AVANT la ligne: if __name__ == '__main__':
# ============================================================================

import random as _random
from scipy import stats as _scipy_stats

def _simulate_line(historical_values):
    """Ligne bookmaker simulée = moyenne pondérée récente arrondie au 0.5"""
    if len(historical_values) < 3:
        return None
    recent = historical_values[-10:]
    weights = np.exp(np.linspace(0, 1, len(recent)))
    weights = weights / weights.sum()
    avg = np.average(recent, weights=weights)
    rounded = round(avg * 2) / 2
    noise = _random.choice([-0.5, 0, 0.5])
    return rounded + noise


def _run_walkforward(player_name, stat_type, season, min_edge, stake):
    """
    Walk-forward backtest pour un joueur/stat.
    Retourne un dict avec tous les résultats.
    """
    STAT_COL_MAP = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}
    VIG_BREAK_EVEN = 52.38
    ODDS = -110
    MIN_TRAIN = 20

    target_col = STAT_COL_MAP.get(stat_type, 'PTS')

    df = collector.get_complete_player_data(player_name, season)
    if df is None or len(df) < MIN_TRAIN + 10:
        return {'status': 'ERROR', 'message': f'Pas assez de données ({len(df) if df is not None else 0} matchs)'}

    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

    # Features
    trend_map = {'PTS': 'recent_trend_pts', 'AST': 'recent_trend_ast',
                 'REB': 'recent_trend_reb', 'FG3M': 'recent_trend_fg3m'}
    avg_map = {
        'PTS': ['avg_pts_last_5', 'avg_pts_last_10'],
        'AST': ['avg_ast_last_5', 'avg_ast_last_10'],
        'REB': ['avg_reb_last_5', 'avg_reb_last_10'],
        'FG3M': ['avg_fg3m_last_5', 'avg_fg3m_last_10']
    }
    candidates = avg_map.get(target_col, []) + [
        'home', 'rest_days', 'minutes_avg',
        'opponent_def_rating', 'pace',
        'usage_rate', 'back_to_back', trend_map.get(target_col, '')
    ]
    feature_cols = [c for c in candidates if c in df.columns]

    if len(feature_cols) < 5:
        return {'status': 'ERROR', 'message': 'Pas assez de features'}

    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error as mse

    _random.seed(42)
    np.random.seed(42)

    bets = []
    all_errors = []

    for i in range(MIN_TRAIN, len(df)):
        df_train = df.iloc[:i].copy()
        X_train = df_train[feature_cols].dropna()
        y_train = df_train.loc[X_train.index, target_col]

        if len(X_train) < 10:
            continue

        actual = float(df.iloc[i][target_col])
        line = _simulate_line(df_train[target_col].values)
        if line is None:
            continue

        try:
            model = xgb.XGBRegressor(
                max_depth=3, learning_rate=0.05, n_estimators=50,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.05, reg_lambda=0.5,
                random_state=42, verbosity=0
            )
            model.fit(X_train, y_train)
        except Exception:
            continue

        y_pred_train = model.predict(X_train)
        train_r2 = float(r2_score(y_train, y_pred_train))
        train_rmse = float(np.sqrt(mse(y_train, y_pred_train)))

        X_pred = df.iloc[i:i+1][feature_cols].fillna(X_train.mean())
        prediction = float(model.predict(X_pred)[0])
        error = abs(actual - prediction)
        all_errors.append(error)

        std = train_rmse if train_rmse > 0 else 3.0
        z = (line - prediction) / std
        over_prob = (1 - _scipy_stats.norm.cdf(z)) * 100
        under_prob = 100 - over_prob

        if over_prob > VIG_BREAK_EVEN:
            edge = over_prob - VIG_BREAK_EVEN
            rec = 'OVER'
            won = actual > line
        elif under_prob > VIG_BREAK_EVEN:
            edge = under_prob - VIG_BREAK_EVEN
            rec = 'UNDER'
            won = actual < line
        else:
            edge = 0
            rec = 'SKIP'
            won = False

        if rec == 'SKIP' or edge < min_edge or train_r2 < 0.05:
            continue

        profit = stake * (100 / abs(ODDS)) if won else -stake

        bets.append({
            'date': str(df.iloc[i]['GAME_DATE'])[:10],
            'actual': round(actual, 1),
            'prediction': round(prediction, 1),
            'line': line,
            'edge': round(edge, 1),
            'recommendation': rec,
            'won': bool(won),
            'profit': round(profit, 2)
        })

    if len(bets) == 0:
        return {
            'status': 'NO_BETS',
            'message': f'0 pari qualifié (edge >= {min_edge}%)',
            'total_predictions': len(all_errors),
            'avg_error': round(float(np.mean(all_errors)), 2) if all_errors else 0
        }

    total_bets = len(bets)
    wins = sum(1 for b in bets if b['won'])
    total_profit = sum(b['profit'] for b in bets)
    roi = total_profit / (total_bets * stake) * 100
    win_rate = wins / total_bets * 100

    return {
        'status': 'SUCCESS',
        'player': player_name,
        'stat_type': stat_type,
        'season': season,
        'total_bets': total_bets,
        'wins': wins,
        'losses': total_bets - wins,
        'win_rate': round(win_rate, 1),
        'total_profit': round(total_profit, 2),
        'roi': round(roi, 2),
        'avg_error': round(float(np.mean(all_errors)), 2) if all_errors else 0,
        'avg_edge': round(float(np.mean([b['edge'] for b in bets])), 1),
        'break_even': round(VIG_BREAK_EVEN, 1),
        'bets': bets[-20:]  # Derniers 20 pour l'affichage
    }


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """
    Lance un walk-forward backtest.
    Body JSON: { player, stat_type, season, min_edge, stake }
    """
    if not XGBOOST_AVAILABLE or not collector:
        return jsonify({'status': 'ERROR', 'message': 'XGBoost non disponible'}), 503

    data = request.get_json()
    player = data.get('player', 'LeBron James')
    stat_type = data.get('stat_type', 'points')
    season = data.get('season', '2023-24')
    min_edge = float(data.get('min_edge', 5.0))
    stake = float(data.get('stake', 50.0))

    print(f"\n🔬 BACKTEST: {player} | {stat_type} | {season} | edge>={min_edge}%")

    try:
        result = _run_walkforward(player, stat_type, season, min_edge, stake)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500
