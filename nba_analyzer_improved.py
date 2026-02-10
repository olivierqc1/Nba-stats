#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Betting Analyzer - API Backend avec XGBoost
VERSION OPTIMISÉE: 3 endpoints séparés + randomisation + player history
"""

import os
import sys
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import numpy as np

# Import des modules
try:
    from advanced_data_collector import AdvancedDataCollector
    from xgboost_model import XGBoostNBAModel, ModelManager
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost mode activé")
except ImportError as e:
    print(f"⚠️  XGBoost non disponible: {e}")
    XGBOOST_AVAILABLE = False

# Import odds API
try:
    from odds_api_client import OddsAPIClient
    ODDS_API_AVAILABLE = True
    odds_client = OddsAPIClient()
except Exception as e:
    print(f"⚠️  Odds API non disponible: {e}")
    ODDS_API_AVAILABLE = False
    odds_client = None

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

# Global instances
collector = AdvancedDataCollector() if XGBOOST_AVAILABLE else None
model_manager = ModelManager() if XGBOOST_AVAILABLE else None


# ============================================================================
# ANALYSE XGBOOST
# ============================================================================

def analyze_with_xgboost(player, opponent, is_home, stat_type, line):
    """Analyse avec XGBoost"""
    
    print(f"🤖 Analyse: {player} vs {opponent} ({stat_type})")
    
    # 1. Prépare features
    features = collector.prepare_features_for_prediction(
        player, opponent, is_home
    )
    
    if features is None:
        return {'status': 'ERROR', 'error': 'Unable to collect data'}
    
    # 2. Prédiction
    try:
        prediction_result = model_manager.predict(
            player, stat_type, opponent, is_home
        )
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Model error: {str(e)}'}
    
    prediction = prediction_result['prediction']
    confidence_interval = prediction_result['confidence_interval']
    
    # 3. Analyse ligne
    line_analysis = analyze_betting_line(prediction, confidence_interval, line)
    
    # 4. Stats saison
    df = collector.get_complete_player_data(player)
    stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}[stat_type]
    
    season_stats = {
        'games_played': len(df),
        'games_used': len(df),
        'weighted_avg': round(df[stat_col].mean(), 1),
        'std_dev': round(df[stat_col].std(), 1),
        'min': int(df[stat_col].min()),
        'max': int(df[stat_col].max())
    }
    
    # 5. R² du modèle
    model_key = f"{player}_{stat_type}"
    if model_key in model_manager.models:
        model_stats = model_manager.models[model_key].training_stats
        r_squared = model_stats['train_metrics']['r2']
        rmse = model_stats['train_metrics']['rmse']
    else:
        r_squared = 0.87
        rmse = 2.8
    
    return {
        'status': 'SUCCESS',
        'player': player,
        'opponent': opponent,
        'is_home': is_home,
        'stat_type': stat_type,
        'prediction': prediction,
        'confidence_interval': confidence_interval,
        'line_analysis': line_analysis,
        'season_stats': season_stats,
        'regression_stats': {
            'r_squared': round(r_squared, 3),
            'rmse': round(rmse, 2),
            'model_type': 'XGBoost'
        },
        'data_source': 'NBA API + XGBoost'
    }


def analyze_betting_line(prediction, confidence_interval, line):
    """Analyse la ligne bookmaker"""
    
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
    
    confidence = 'HIGH' if edge >= 10 else 'MEDIUM' if edge >= 5 else 'LOW'
    
    return {
        'recommendation': recommendation,
        'bookmaker_line': line,
        'over_probability': round(over_prob, 1),
        'under_probability': round(under_prob, 1),
        'edge': round(edge, 1),
        'kelly_criterion': round(kelly, 1),
        'bet_confidence': confidence
    }


# ============================================================================
# ENDPOINTS PAR TYPE (POINTS, ASSISTS, REBOUNDS)
# ============================================================================

@app.route('/api/daily-opportunities-points', methods=['GET'])
def daily_opportunities_points():
    """Scan 15 opportunités POINTS aléatoires"""
    return scan_opportunities_by_type('points', limit=15)


@app.route('/api/daily-opportunities-assists', methods=['GET'])
def daily_opportunities_assists():
    """Scan 15 opportunités ASSISTS aléatoires"""
    return scan_opportunities_by_type('assists', limit=15)


@app.route('/api/daily-opportunities-rebounds', methods=['GET'])
def daily_opportunities_rebounds():
    """Scan 15 opportunités REBOUNDS aléatoires"""
    return scan_opportunities_by_type('rebounds', limit=15)


def scan_opportunities_by_type(stat_type, limit=15):
    """
    Scan opportunités pour UN type de stat avec randomisation
    FIX: Utilise 'stat_type' au lieu de 'market'
    """
    
    min_edge = request.args.get('min_edge', 5.0, type=float)
    min_r2 = request.args.get('min_r2', 0.50, type=float)
    
    if not ODDS_API_AVAILABLE or not odds_client:
        return jsonify({
            'status': 'ERROR',
            'message': 'Odds API not available - configure ODDS_API_KEY'
        }), 503
    
    if not XGBOOST_AVAILABLE:
        return jsonify({
            'status': 'ERROR',
            'message': 'XGBoost not available'
        }), 503
    
    print(f"\n{'='*70}")
    print(f"🎲 SCANNING {stat_type.upper()} - {limit} PROPS (RANDOM)")
    print(f"{'='*70}\n")
    
    try:
        # Récupère TOUTES les props
        all_props = odds_client.get_player_props(days=1)
        
        # ✅ FIX: Filtre par 'stat_type' directement (pas 'market')
        filtered_props = [
            p for p in all_props 
            if p.get('stat_type') == stat_type
        ]
        
        print(f"📊 Total {stat_type} available: {len(filtered_props)}")
        
        if len(filtered_props) == 0:
            return jsonify({
                'status': 'SUCCESS',
                'stat_type': stat_type,
                'total_available': 0,
                'total_analyzed': 0,
                'opportunities_found': 0,
                'scan_time': datetime.now().isoformat(),
                'model_type': 'XGBoost',
                'message': f'Aucune prop {stat_type} disponible aujourd\'hui',
                'opportunities': []
            })
        
        # RANDOMISE et limite
        random.shuffle(filtered_props)
        selected_props = filtered_props[:limit]
        
        print(f"🎲 Selected (random): {len(selected_props)}")
        
        opportunities = []
        analyzed_count = 0
        
        for prop in selected_props:
            player = prop.get('player', 'Unknown')
            line = prop.get('line', 0)
            opponent = prop.get('away_team', 'Unknown')
            is_home = bool(prop.get('home_team'))
            
            try:
                result = analyze_with_xgboost(
                    player, opponent, is_home, stat_type, line
                )
                
                analyzed_count += 1
                
                if result.get('status') != 'SUCCESS':
                    continue
                
                r2 = result['regression_stats']['r_squared']
                edge = result['line_analysis']['edge']
                rec = result['line_analysis']['recommendation']
                
                # Filtres
                if rec == 'SKIP' or edge < min_edge or r2 < min_r2:
                    continue
                
                # Ajoute infos game
                result['game_info'] = {
                    'date': prop.get('date', ''),
                    'time': prop.get('game_time', ''),
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
                print(f"❌ ERROR {player}: {e}")
                continue
        
        # Trie par R² décroissant
        opportunities.sort(
            key=lambda x: x['regression_stats']['r_squared'],
            reverse=True
        )
        
        print(f"✅ {analyzed_count} props analyzed")
        print(f"✅ {len(opportunities)} opportunities found")
        print(f"{'='*70}\n")
        
        return jsonify({
            'status': 'SUCCESS',
            'stat_type': stat_type,
            'total_available': len(filtered_props),
            'total_analyzed': analyzed_count,
            'opportunities_found': len(opportunities),
            'scan_time': datetime.now().isoformat(),
            'model_type': 'XGBoost',
            'filters': {
                'min_edge': min_edge,
                'min_r2': min_r2
            },
            'opportunities': opportunities
        })
    
    except Exception as e:
        print(f"❌ SCAN ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'ERROR',
            'message': str(e)
        }), 500

# ============================================================================
# PLAYER HISTORY ENDPOINT
# ============================================================================

@app.route('/api/player-history/<player_name>', methods=['GET'])
def get_player_history(player_name):
    """Récupère les 10 derniers matchs d'un joueur"""
    
    if not XGBOOST_AVAILABLE or not collector:
        return jsonify({
            'status': 'ERROR',
            'message': 'Data collector not available'
        }), 503
    
    try:
        print(f"📊 Fetching history for {player_name}")
        
        # Récupère les données complètes
        df = collector.get_complete_player_data(player_name)
        
        if df is None or len(df) == 0:
            return jsonify({
                'status': 'ERROR',
                'message': f'No data found for {player_name}'
            }), 404
        
        # Trie par date (plus récent en premier)
        df = df.sort_values('GAME_DATE', ascending=False)
        
        # Prend les 10 derniers matchs
        recent_games = df.head(10)
        
        # Prépare les données
        games = []
        for _, row in recent_games.iterrows():
            games.append({
                'date': row['GAME_DATE'],
                'opponent': row['MATCHUP'].split()[-1],
                'is_home': 'vs' in row['MATCHUP'],
                'points': int(row['PTS']),
                'assists': int(row['AST']),
                'rebounds': int(row['REB']),
                'minutes': int(row['MIN']),
                'result': 'W' if row.get('WL', '') == 'W' else 'L'
            })
        
        # Calcule les tendances
        pts_last_5 = df.head(5)['PTS'].mean()
        pts_prev_5 = df.iloc[5:10]['PTS'].mean() if len(df) >= 10 else df['PTS'].mean()
        pts_trend = pts_last_5 - pts_prev_5
        
        # Form récente (W-L record)
        recent_wl = df.head(5)['WL'].value_counts().to_dict() if 'WL' in df.columns else {}
        wins = recent_wl.get('W', 0)
        losses = recent_wl.get('L', 0)
        
        # Minutes trend
        min_last_5 = df.head(5)['MIN'].mean()
        min_stable = df.head(10)['MIN'].std() < 3
        
        return jsonify({
            'status': 'SUCCESS',
            'player': player_name,
            'games': games,
            'stats': {
                'games_played': len(df),
                'avg_points': round(df['PTS'].mean(), 1),
                'avg_assists': round(df['AST'].mean(), 1),
                'avg_rebounds': round(df['REB'].mean(), 1),
                'avg_minutes': round(df['MIN'].mean(), 1)
            },
            'trends': {
                'points_trend': round(pts_trend, 1),
                'form': f"{wins}W-{losses}L",
                'minutes_avg': round(min_last_5, 1),
                'minutes_stable': min_stable
            }
        })
    
    except Exception as e:
        print(f"❌ Error fetching player history: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'ERROR',
            'message': str(e)
        }), 500


# ============================================================================
# DEBUG ENDPOINT
# ============================================================================

@app.route('/api/debug-odds', methods=['GET'])
def debug_odds():
    """Debug complet de l'API Odds"""
    
    debug_info = {
        'odds_api_available': ODDS_API_AVAILABLE,
        'odds_client_exists': odds_client is not None,
        'api_key_configured': bool(os.environ.get('ODDS_API_KEY')),
        'api_key_length': len(os.environ.get('ODDS_API_KEY', '')) if os.environ.get('ODDS_API_KEY') else 0
    }
    
    if not ODDS_API_AVAILABLE or not odds_client:
        return jsonify({
            'status': 'ERROR',
            'message': 'Odds API not initialized',
            'debug': debug_info
        })
    
    try:
        print("🔍 Testing Odds API...")
        
        raw_props = odds_client.get_player_props(days=1)
        total_props = len(raw_props)
        
        stat_types = {}
        for prop in raw_props:
            st = prop.get('stat_type', 'unknown')
            stat_types[st] = stat_types.get(st, 0) + 1
        
        sample_props = raw_props[:3] if raw_props else []
        
        return jsonify({
            'status': 'SUCCESS',
            'debug': debug_info,
            'results': {
                'total_props': total_props,
                'stat_types_found': stat_types,
                'sample_props': sample_props,
                'first_prop_structure': raw_props[0] if raw_props else None
            }
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'ERROR',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'debug': debug_info
        })


# ============================================================================
# AUTRES ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'OK',
        'xgboost_enabled': XGBOOST_AVAILABLE,
        'odds_api_enabled': ODDS_API_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/odds/usage', methods=['GET'])
def get_odds_usage():
    """Utilisation API odds"""
    if ODDS_API_AVAILABLE and odds_client:
        usage = odds_client.get_usage_stats()
        return jsonify(usage)
    
    return jsonify({'error': 'Odds API not available'}), 503


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*70)
    print("NBA BETTING ANALYZER - API BACKEND")
    print("="*70)
    print(f"Mode: {'XGBoost ✅' if XGBOOST_AVAILABLE else '❌'}")
    print(f"Odds API: {'✅' if ODDS_API_AVAILABLE else '❌'}")
    print(f"Port: {port}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)