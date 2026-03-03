#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Betting Analyzer - API Backend avec XGBoost
VERSION RENDER PAYANT: 4 endpoints (PTS/AST/REB/3PT) + backtest integre
"""

import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import numpy as np

try:
    from advanced_data_collector import AdvancedDataCollector
    from xgboost_model import XGBoostNBAModel, ModelManager
    XGBOOST_AVAILABLE = True
    print("XGBoost mode active")
except ImportError as e:
    print(f"XGBoost non disponible: {e}")
    XGBOOST_AVAILABLE = False

try:
    from odds_api_client import OddsAPIClient
    ODDS_API_AVAILABLE = True
    odds_client = OddsAPIClient()
except Exception as e:
    print(f"Odds API non disponible: {e}")
    ODDS_API_AVAILABLE = False
    odds_client = None

app = Flask(__name__)
CORS(app)

collector = AdvancedDataCollector() if XGBOOST_AVAILABLE else None
model_manager = ModelManager() if XGBOOST_AVAILABLE else None


def analyze_betting_line(prediction, confidence_interval, line):
    if line is None:
        return {'recommendation': 'NO_LINE', 'bookmaker_line': None}
    std = (confidence_interval['upper'] - confidence_interval['lower']) / 3.92
    from scipy import stats
    z_score = (line - prediction) / std
    over_prob = (1 - stats.norm.cdf(z_score)) * 100
    under_prob = 100 - over_prob
    implied_prob = 52.4
    if over_prob > implied_prob:
        edge = over_prob - implied_prob
        recommendation = 'OVER'
        bet_prob = over_prob
    elif under_prob > implied_prob:
        edge = under_prob - implied_prob
        recommendation = 'UNDER'
        bet_prob = under_prob
    else:
        edge = 0
        recommendation = 'SKIP'
        bet_prob = max(over_prob, under_prob)
    kelly = (bet_prob/100 - (1-bet_prob/100)) * 100 if edge > 5 else 0
    kelly = max(min(kelly, 10), 0)
    return {
        'recommendation': recommendation,
        'bookmaker_line': line,
        'over_probability': round(over_prob, 1),
        'under_probability': round(under_prob, 1),
        'edge': round(edge, 1),
        'kelly_criterion': round(kelly, 1),
        'bet_confidence': 'HIGH' if edge >= 10 else 'MEDIUM' if edge >= 5 else 'LOW'
    }


def analyze_with_xgboost(player, opponent, is_home, stat_type, line):
    print(f"Analyse: {player} vs {opponent} ({stat_type})")
    features = collector.prepare_features_for_prediction(player, opponent, is_home)
    if features is None:
        return {'status': 'ERROR', 'error': 'Unable to collect data'}
    try:
        prediction_result = model_manager.predict(player, stat_type, opponent, is_home)
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Model error: {str(e)}'}
    prediction = prediction_result['prediction']
    confidence_interval = prediction_result['confidence_interval']
    explanation = prediction_result.get('explanation')
    line_analysis = analyze_betting_line(prediction, confidence_interval, line)
    df = collector.get_complete_player_data(player)
    stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}[stat_type]
    season_stats = {
        'games_played': len(df), 'games_used': len(df),
        'weighted_avg': round(df[stat_col].mean(), 1),
        'std_dev': round(df[stat_col].std(), 1),
        'min': int(df[stat_col].min()), 'max': int(df[stat_col].max())
    }
    model_key = f"{player}_{stat_type}"
    if model_key in model_manager.models:
        model_stats = model_manager.models[model_key].training_stats
        r_squared = model_stats.get('test_metrics', {}).get('r2', 0.5)
        rmse = model_stats.get('test_metrics', {}).get('rmse', 5.0)
    else:
        r_squared = 0.50
        rmse = 5.0
    response = {
        'status': 'SUCCESS', 'player': player, 'opponent': opponent,
        'is_home': is_home, 'stat_type': stat_type, 'prediction': prediction,
        'confidence_interval': confidence_interval, 'line_analysis': line_analysis,
        'season_stats': season_stats,
        'regression_stats': {
            'r_squared': round(r_squared, 3),
            'rmse': round(rmse, 2),
            'model_type': 'XGBoost'
        },
        'data_source': 'NBA API + XGBoost'
    }
    if explanation is not None:
        response['explanation'] = explanation
    return response


@app.route('/api/daily-opportunities-points', methods=['GET'])
def daily_opportunities_points():
    return scan_opportunities_by_type('points', limit=25)

@app.route('/api/daily-opportunities-assists', methods=['GET'])
def daily_opportunities_assists():
    return scan_opportunities_by_type('assists', limit=25)

@app.route('/api/daily-opportunities-rebounds', methods=['GET'])
def daily_opportunities_rebounds():
    return scan_opportunities_by_type('rebounds', limit=25)

@app.route('/api/daily-opportunities-3pt', methods=['GET'])
def daily_opportunities_3pt():
    return scan_opportunities_by_type('3pt', limit=25)


def scan_opportunities_by_type(stat_type, limit=25):
    min_edge = request.args.get('min_edge', 5.0, type=float)
    min_r2 = request.args.get('min_r2', 0.50, type=float)
    if not ODDS_API_AVAILABLE or not odds_client:
        return jsonify({'status': 'ERROR', 'message': 'Odds API not available'}), 503
    if not XGBOOST_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'XGBoost not available'}), 503
    try:
        all_props = odds_client.get_player_props(days=1)
        filtered_props = [p for p in all_props if p.get('stat_type') == stat_type]
        if len(filtered_props) == 0:
            return jsonify({
                'status': 'SUCCESS', 'stat_type': stat_type,
                'total_available': 0, 'total_analyzed': 0,
                'opportunities_found': 0,
                'scan_time': datetime.now().isoformat(),
                'model_type': 'XGBoost',
                'message': f'Aucune prop {stat_type} disponible',
                'opportunities': []
            })
        random.shuffle(filtered_props)
        selected_props = filtered_props[:limit]
        opportunities = []
        analyzed_count = 0
        for prop in selected_props:
            player = prop.get('player', 'Unknown')
            line = prop.get('line', 0)
            opponent = prop.get('away_team', 'Unknown')
            is_home = bool(prop.get('home_team'))
            try:
                result = analyze_with_xgboost(player, opponent, is_home, stat_type, line)
                analyzed_count += 1
                if result.get('status') != 'SUCCESS':
                    continue
                r2 = result['regression_stats']['r_squared']
                edge = result['line_analysis']['edge']
                rec = result['line_analysis']['recommendation']
                if rec == 'SKIP' or edge < min_edge or r2 < min_r2:
                    continue
                result['game_info'] = {
                    'date': prop.get('date', ''),
                    'home_team': prop.get('home_team', ''),
                    'away_team': prop.get('away_team', '')
                }
                result['bookmaker_info'] = {
                    'bookmaker': prop.get('bookmaker', 'Unknown'),
                    'line': line,
                    'over_odds': prop.get('over_odds', -110),
                    'under_odds': prop.get('under_odds', -110)
                }
                opportunities.append(result)
            except Exception as e:
                print(f"ERROR {player}: {e}")
                continue
        opportunities.sort(key=lambda x: x['regression_stats']['r_squared'], reverse=True)
        return jsonify({
            'status': 'SUCCESS', 'stat_type': stat_type,
            'total_available': len(filtered_props),
            'total_analyzed': analyzed_count,
            'opportunities_found': len(opportunities),
            'scan_time': datetime.now().isoformat(),
            'model_type': 'XGBoost', 'render_tier': 'PAID',
            'filters': {'min_edge': min_edge, 'min_r2': min_r2},
            'opportunities': opportunities
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500

@app.route('/api/player-history/<player_name>', methods=['GET'])
def get_player_history(player_name):
    if not XGBOOST_AVAILABLE or not collector:
        return jsonify({'status': 'ERROR', 'message': 'Data collector not available'}), 503
    try:
        df = collector.get_complete_player_data(player_name)
        if df is None or len(df) == 0:
            return jsonify({'status': 'ERROR', 'message': f'No data found for {player_name}'}), 404
        df = df.sort_values('GAME_DATE', ascending=False)
        games = []
        for _, row in df.head(10).iterrows():
            games.append({
                'date': str(row['GAME_DATE'])[:10],
                'opponent': row['MATCHUP'].split()[-1],
                'is_home': 'vs' in row['MATCHUP'],
                'points': int(row['PTS']),
                'assists': int(row['AST']),
                'rebounds': int(row['REB']),
                'three_pointers': int(row.get('FG3M', 0)),
                'minutes': int(row['MIN']),
                'result': 'W' if row.get('WL', '') == 'W' else 'L'
            })
        recent_wl = df.head(5)['WL'].value_counts().to_dict() if 'WL' in df.columns else {}
        pts_trend = 0
        if len(df) >= 10:
            pts_trend = round(df.head(5)['PTS'].mean() - df.iloc[5:10]['PTS'].mean(), 1)
        return jsonify({
            'status': 'SUCCESS', 'player': player_name, 'games': games,
            'stats': {
                'games_played': len(df),
                'avg_points': round(df['PTS'].mean(), 1),
                'avg_assists': round(df['AST'].mean(), 1),
                'avg_rebounds': round(df['REB'].mean(), 1),
                'avg_minutes': round(df['MIN'].mean(), 1)
            },
            'trends': {
                'points_trend': pts_trend,
                'form': f"{recent_wl.get('W',0)}W-{recent_wl.get('L',0)}L",
                'minutes_avg': round(df.head(5)['MIN'].mean(), 1)
            }
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'xgboost_enabled': XGBOOST_AVAILABLE,
        'odds_api_enabled': ODDS_API_AVAILABLE,
        'render_tier': 'PAID',
        'backtest_enabled': True,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/odds/usage', methods=['GET'])
def get_odds_usage():
    if ODDS_API_AVAILABLE and odds_client:
        usage = odds_client.get_usage_stats()
        return jsonify(usage)
    return jsonify({'error': 'Odds API not available'}), 503


# ============================================================================
# BACKTEST WALK-FORWARD
# ============================================================================

import random as _random
from scipy import stats as _scipy_stats


def _simulate_line(historical_values):
    """Simule une ligne bookmaker via moyenne pondérée récente."""
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
    """Walk-forward backtest — zero data leakage."""
    STAT_COL_MAP = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}
    VIG_BREAK_EVEN = 52.38
    ODDS = -110
    MIN_TRAIN = 20

    target_col = STAT_COL_MAP.get(stat_type, 'PTS')

    df = collector.get_complete_player_data(player_name, season)
    if df is None or len(df) < MIN_TRAIN + 10:
        n = len(df) if df is not None else 0
        return {'status': 'ERROR', 'message': f'Pas assez de donnees ({n} matchs, besoin de {MIN_TRAIN + 10})'}

    df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

    trend_map = {
        'PTS': 'recent_trend_pts', 'AST': 'recent_trend_ast',
        'REB': 'recent_trend_reb', 'FG3M': 'recent_trend_fg3m'
    }
    avg_map = {
        'PTS':  ['avg_pts_last_5',  'avg_pts_last_10'],
        'AST':  ['avg_ast_last_5',  'avg_ast_last_10'],
        'REB':  ['avg_reb_last_5',  'avg_reb_last_10'],
        'FG3M': ['avg_fg3m_last_5', 'avg_fg3m_last_10']
    }
    candidates = avg_map.get(target_col, []) + [
        'home', 'rest_days', 'minutes_avg',
        'opponent_def_rating', 'pace',
        'usage_rate', 'back_to_back',
        trend_map.get(target_col, '')
    ]
    feature_cols = [c for c in candidates if c in df.columns]

    if len(feature_cols) < 5:
        return {'status': 'ERROR', 'message': f'Pas assez de features ({len(feature_cols)} trouvees)'}

    import xgboost as xgb
    from sklearn.metrics import r2_score, mean_squared_error as mse_fn

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
        train_r2   = float(r2_score(y_train, y_pred_train))
        train_rmse = float(np.sqrt(mse_fn(y_train, y_pred_train)))

        X_pred = df.iloc[i:i+1][feature_cols].fillna(X_train.mean())
        prediction = float(model.predict(X_pred)[0])
        all_errors.append(abs(actual - prediction))

        std = train_rmse if train_rmse > 0 else 3.0
        z = (line - prediction) / std
        over_prob  = (1 - _scipy_stats.norm.cdf(z)) * 100
        under_prob = 100 - over_prob

        if over_prob > VIG_BREAK_EVEN:
            edge = over_prob - VIG_BREAK_EVEN
            rec  = 'OVER'
            won  = actual > line
        elif under_prob > VIG_BREAK_EVEN:
            edge = under_prob - VIG_BREAK_EVEN
            rec  = 'UNDER'
            won  = actual < line
        else:
            edge = 0
            rec  = 'SKIP'
            won  = False

        if rec == 'SKIP' or edge < min_edge or train_r2 < 0.05:
            continue

        profit = stake * (100 / abs(ODDS)) if won else -stake
        bets.append({
            'date':           str(df.iloc[i]['GAME_DATE'])[:10],
            'actual':         round(actual, 1),
            'prediction':     round(prediction, 1),
            'line':           line,
            'edge':           round(edge, 1),
            'recommendation': rec,
            'won':            bool(won),
            'profit':         round(profit, 2)
        })

    if len(bets) == 0:
        return {
            'status':            'NO_BETS',
            'message':           f'0 pari qualifie (edge >= {min_edge}%)',
            'total_predictions': len(all_errors),
            'avg_error':         round(float(np.mean(all_errors)), 2) if all_errors else 0
        }

    total_bets   = len(bets)
    wins         = sum(1 for b in bets if b['won'])
    total_profit = sum(b['profit'] for b in bets)
    roi          = total_profit / (total_bets * stake) * 100
    win_rate     = wins / total_bets * 100

    return {
        'status':       'SUCCESS',
        'player':       player_name,
        'stat_type':    stat_type,
        'season':       season,
        'total_bets':   total_bets,
        'wins':         wins,
        'losses':       total_bets - wins,
        'win_rate':     round(win_rate, 1),
        'total_profit': round(total_profit, 2),
        'roi':          round(roi, 2),
        'avg_error':    round(float(np.mean(all_errors)), 2) if all_errors else 0,
        'avg_edge':     round(float(np.mean([b['edge'] for b in bets])), 1),
        'break_even':   round(VIG_BREAK_EVEN, 1),
        'bets':         bets[-20:]
    }


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Walk-forward backtest sur une saison historique complète."""
    if not XGBOOST_AVAILABLE or not collector:
        return jsonify({'status': 'ERROR', 'message': 'XGBoost non disponible'}), 503

    data      = request.get_json()
    player    = data.get('player',    'LeBron James')
    stat_type = data.get('stat_type', 'points')
    season    = data.get('season',    '2023-24')
    min_edge  = float(data.get('min_edge', 5.0))
    stake     = float(data.get('stake',    50.0))

    print(f"\n BACKTEST: {player} | {stat_type} | {season} | edge>={min_edge}%")

    try:
        result = _run_walkforward(player, stat_type, season, min_edge, stake)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    print("\n" + "="*70)
    print("NBA BETTING ANALYZER - API BACKEND")
    print("="*70)
    print(f"XGBoost:  {'OK' if XGBOOST_AVAILABLE else 'NON DISPO'}")
    print(f"Odds API: {'OK' if ODDS_API_AVAILABLE else 'NON DISPO'}")
    print(f"Backtest: OK (endpoint /api/backtest)")
    print(f"Port:     {port}")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
