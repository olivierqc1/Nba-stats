#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Analyzer - Fichier principal
Endpoints: daily opportunities (4 types), player history, health, odds usage
Le backtest est dans nba_analyzer_backtest.py
"""

import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import numpy as np

try:
    from advanced_data_collector import AdvancedDataCollector
    from xgboost_model import ModelManager
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

collector     = AdvancedDataCollector() if XGBOOST_AVAILABLE else None
model_manager = ModelManager()          if XGBOOST_AVAILABLE else None


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def analyze_betting_line(prediction, confidence_interval, line):
    if line is None:
        return {'recommendation': 'NO_LINE', 'bookmaker_line': None}
    from scipy import stats
    std  = max((confidence_interval['upper'] - confidence_interval['lower']) / 3.92, 1.0)
    z    = (line - prediction) / std
    op   = (1 - stats.norm.cdf(z)) * 100
    up   = 100 - op
    if op > 52.4:
        edge, rec, bp = op - 52.4, 'OVER',  op
    elif up > 52.4:
        edge, rec, bp = up - 52.4, 'UNDER', up
    else:
        edge, rec, bp = 0, 'SKIP', max(op, up)
    kelly = max(min((bp / 100 - (1 - bp / 100)) * 100, 10), 0) if edge > 5 else 0
    return {
        'recommendation':    rec,
        'bookmaker_line':    line,
        'over_probability':  round(op,    1),
        'under_probability': round(up,    1),
        'edge':              round(edge,  1),
        'kelly_criterion':   round(kelly, 1),
        'bet_confidence':    'HIGH' if edge >= 10 else 'MEDIUM' if edge >= 5 else 'LOW'
    }


def analyze_with_xgboost(player, opponent, is_home, stat_type, line):
    features = collector.prepare_features_for_prediction(player, opponent, is_home)
    if features is None:
        return {'status': 'ERROR', 'error': 'Unable to collect data'}
    try:
        pred_result = model_manager.predict(player, stat_type, opponent, is_home)
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}
    if pred_result.get('prediction') is None:
        return {'status': 'ERROR', 'error': pred_result.get('error', 'Unknown')}

    prediction = pred_result['prediction']
    ci         = pred_result['confidence_interval']

    line_value = 0.0
    if line is not None:
        try:
            line_value = collector.get_line_value_score(player, stat_type, line)
        except Exception:
            pass

    line_analysis = analyze_betting_line(prediction, ci, line)
    line_analysis['line_value_score'] = round(line_value, 2)

    df       = collector.get_complete_player_data(player)
    stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}[stat_type]
    excluded = int(df['games_excluded_count'].iloc[0]) if 'games_excluded_count' in df.columns else 0

    season_stats = {
        'games_played':    len(df),
        'games_excluded':  excluded,
        'weighted_avg':    round(float(df[stat_col].mean()),       1),
        'std_dev':         round(float(df[stat_col].std()),        1),
        'last_5_avg':      round(float(df[stat_col].head(5).mean()), 1),
        'last_10_avg':     round(float(df[stat_col].head(10).mean()), 1),
        'min':             int(df[stat_col].min()),
        'max':             int(df[stat_col].max())
    }

    key = f"{player}_{stat_type}"
    if key in model_manager.models:
        ms      = model_manager.models[key].training_stats
        r2      = ms.get('test_metrics', {}).get('r2',   0.5)
        rmse    = ms.get('test_metrics', {}).get('rmse', 5.0)
        overfit = ms.get('overfit_gap', 0.0)
    else:
        r2, rmse, overfit = 0.5, 5.0, 0.0

    return {
        'status': 'SUCCESS', 'player': player, 'opponent': opponent,
        'is_home': is_home, 'stat_type': stat_type,
        'prediction': prediction, 'confidence_interval': ci,
        'line_analysis': line_analysis, 'season_stats': season_stats,
        'regression_stats': {
            'r_squared':   round(r2,      3),
            'rmse':        round(rmse,    2),
            'overfit_gap': round(overfit, 3),
            'model_type':  'XGBoost'
        },
        'data_source': 'NBA API + XGBoost v3'
    }


# ------------------------------------------------------------------
# DAILY OPPORTUNITIES
# ------------------------------------------------------------------

@app.route('/api/daily-opportunities-points',   methods=['GET'])
def daily_opportunities_points():
    return scan_by_type('points',   25)

@app.route('/api/daily-opportunities-assists',  methods=['GET'])
def daily_opportunities_assists():
    return scan_by_type('assists',  25)

@app.route('/api/daily-opportunities-rebounds', methods=['GET'])
def daily_opportunities_rebounds():
    return scan_by_type('rebounds', 25)

@app.route('/api/daily-opportunities-3pt',      methods=['GET'])
def daily_opportunities_3pt():
    return scan_by_type('3pt',      25)


def scan_by_type(stat_type, limit=25):
    min_edge       = request.args.get('min_edge',       5.0,  type=float)
    min_r2         = request.args.get('min_r2',         0.40, type=float)
    min_line_value = request.args.get('min_line_value', 0.5,  type=float)

    if not ODDS_API_AVAILABLE or not XGBOOST_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Service unavailable'}), 503

    try:
        props = [p for p in odds_client.get_player_props(days=1) if p.get('stat_type') == stat_type]
        if not props:
            return jsonify({
                'status': 'SUCCESS', 'stat_type': stat_type,
                'total_available': 0, 'total_analyzed': 0, 'opportunities_found': 0,
                'scan_time': datetime.now().isoformat(), 'opportunities': []
            })

        random.shuffle(props)
        opportunities  = []
        analyzed_count = 0

        for prop in props[:limit]:
            player   = prop.get('player', 'Unknown')
            line     = prop.get('line', 0)
            opponent = prop.get('away_team', 'Unknown')
            is_home  = bool(prop.get('home_team'))
            try:
                result = analyze_with_xgboost(player, opponent, is_home, stat_type, line)
                analyzed_count += 1
                if result.get('status') != 'SUCCESS':
                    continue
                r2         = result['regression_stats']['r_squared']
                overfit    = result['regression_stats']['overfit_gap']
                rmse       = result['regression_stats']['rmse']
                edge       = result['line_analysis']['edge']
                rec        = result['line_analysis']['recommendation']
                lv         = abs(result['line_analysis'].get('line_value_score', 0))
                rmse_ratio = (rmse / line) if line and line > 0 else 1.0

                # Filtre RMSE/ligne > 40%: modele trop imprécis pour ce joueur/ligne
                if rec == 'SKIP' or edge < min_edge or r2 < min_r2 or overfit > 0.35 or lv < min_line_value or rmse_ratio > 0.40:
                    continue

                result['regression_stats']['rmse_ratio'] = round(rmse_ratio, 2)
                result['game_info']      = {'date': prop.get('date',''), 'home_team': prop.get('home_team',''), 'away_team': prop.get('away_team','')}
                result['bookmaker_info'] = {'bookmaker': prop.get('bookmaker','Unknown'), 'line': line, 'over_odds': prop.get('over_odds',-110), 'under_odds': prop.get('under_odds',-110)}
                opportunities.append(result)
            except Exception as e:
                print(f"ERROR {player}: {e}")

        opportunities.sort(key=lambda x: (abs(x['line_analysis'].get('line_value_score',0)), x['regression_stats']['r_squared']), reverse=True)

        return jsonify({
            'status': 'SUCCESS', 'stat_type': stat_type,
            'total_available': len(props), 'total_analyzed': analyzed_count,
            'opportunities_found': len(opportunities),
            'scan_time': datetime.now().isoformat(),
            'filters': {'min_edge': min_edge, 'min_r2': min_r2, 'min_line_value': min_line_value},
            'opportunities': opportunities
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# ------------------------------------------------------------------
# PLAYER HISTORY
# ------------------------------------------------------------------

@app.route('/api/player-history/<player_name>', methods=['GET'])
def get_player_history(player_name):
    if not XGBOOST_AVAILABLE:
        return jsonify({'status': 'ERROR', 'message': 'Not available'}), 503
    try:
        df = collector.get_complete_player_data(player_name)
        if df is None or len(df) == 0:
            return jsonify({'status': 'ERROR', 'message': f'No data for {player_name}'}), 404

        df      = df.sort_values('GAME_DATE', ascending=False)
        games   = []
        excluded = int(df['games_excluded_count'].iloc[0]) if 'games_excluded_count' in df.columns else 0

        for _, row in df.head(10).iterrows():
            games.append({
                'date':           str(row['GAME_DATE'])[:10],
                'opponent':       row['MATCHUP'].split()[-1],
                'is_home':        'vs' in row['MATCHUP'],
                'points':         int(row['PTS']),
                'assists':        int(row['AST']),
                'rebounds':       int(row['REB']),
                'three_pointers': int(row.get('FG3M', 0)),
                'minutes':        int(row['MIN']),
                'result':         'W' if row.get('WL','') == 'W' else 'L'
            })

        wl = df.head(5)['WL'].value_counts().to_dict() if 'WL' in df.columns else {}
        return jsonify({
            'status': 'SUCCESS', 'player': player_name, 'games': games,
            'games_excluded': excluded,
            'stats': {
                'games_played': len(df), 'avg_points': round(float(df['PTS'].mean()),1),
                'avg_assists':  round(float(df['AST'].mean()),1),
                'avg_rebounds': round(float(df['REB'].mean()),1),
                'avg_minutes':  round(float(df['MIN'].mean()),1),
                'last_5_pts':   round(float(df['PTS'].head(5).mean()),1)
            },
            'trends': {
                'points_trend': round(float(df['PTS'].head(5).mean() - df['PTS'].iloc[5:10].mean()), 1) if len(df) >= 10 else 0,
                'form':        f"{wl.get('W',0)}W-{wl.get('L',0)}L",
                'minutes_avg': round(float(df['MIN'].head(5).mean()), 1)
            }
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'status': 'ERROR', 'message': str(e)}), 500


# ------------------------------------------------------------------
# HEALTH + USAGE
# ------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK', 'version': 'v3-split',
        'xgboost_enabled': XGBOOST_AVAILABLE,
        'odds_api_enabled': ODDS_API_AVAILABLE,
        'outlier_filter': True,
        'backtest_enabled': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/odds/usage', methods=['GET'])
def get_odds_usage():
    if ODDS_API_AVAILABLE and odds_client:
        return jsonify(odds_client.get_usage_stats())
    return jsonify({'error': 'Odds API not available'}), 503


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

# Enregistrement du backtest au niveau module (requis pour gunicorn/Render)
try:
    from nba_analyzer_backtest import register_backtest_routes
    register_backtest_routes(app, collector)
except Exception as _e:
    print(f"Backtest module non charge: {_e}")

if __name__ == '__main__':

    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    print("\n" + "="*55)
    print("NBA ANALYZER v3 — SPLIT VERSION")
    print(f"XGBoost:  {'OK' if XGBOOST_AVAILABLE else 'NON DISPO'}")
    print(f"Odds API: {'OK' if ODDS_API_AVAILABLE else 'NON DISPO'}")
    print(f"Outlier filter: ACTIF (< 20 min + winsorisation 2.5σ)")
    print(f"Line value filter: ACTIF (min=0.5)")
    print(f"Backtest: nba_analyzer_backtest.py")
    print(f"Port: {port}")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=port, debug=debug)
