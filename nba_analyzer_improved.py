#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Betting Analyzer - API Backend avec XGBoost
VERSION RENDER PAYANT: 4 endpoints (PTS/AST/REB/3PT) + 50 props par type
PARTIE 1/2
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
# ANALYSE XGBOOST AVEC SHAP EXPLANATIONS
# ============================================================================

def analyze_with_xgboost(player, opponent, is_home, stat_type, line):
    """
    Analyse avec XGBoost + SHAP explanations
    """
    
    print(f"🤖 Analyse: {player} vs {opponent} ({stat_type})")
    
    # 1. Prépare features
    features = collector.prepare_features_for_prediction(
        player, opponent, is_home
    )
    
    if features is None:
        return {'status': 'ERROR', 'error': 'Unable to collect data'}
    
    # 2. Prédiction AVEC EXPLICATION
    try:
        prediction_result = model_manager.predict(
            player, stat_type, opponent, is_home
        )
    except Exception as e:
        return {'status': 'ERROR', 'error': f'Model error: {str(e)}'}
    
    prediction = prediction_result['prediction']
    confidence_interval = prediction_result['confidence_interval']
    explanation = prediction_result.get('explanation')
    
    # 3. Analyse ligne
    line_analysis = analyze_betting_line(prediction, confidence_interval, line)
    
    # 4. Stats saison
    df = collector.get_complete_player_data(player)
    stat_col = {
        'points': 'PTS',
        'assists': 'AST',
        'rebounds': 'REB',
        '3pt': 'FG3M'  # ← NOUVEAU: 3-pointers made
    }[stat_type]
    
    season_stats = {
        'games_played': len(df),
        'games_used': len(df),
        'weighted_avg': round(df[stat_col].mean(), 1),
        'std_dev': round(df[stat_col].std(), 1),
        'min': int(df[stat_col].min()),
        'max': int(df[stat_col].max())
    }
    
    # 5. R² du modèle (TEST R², pas train!)
    model_key = f"{player}_{stat_type}"
    if model_key in model_manager.models:
        model_stats = model_manager.models[model_key].training_stats
        r_squared = model_stats.get('test_metrics', {}).get('r2', 0.5)
        rmse = model_stats.get('test_metrics', {}).get('rmse', 5.0)
    else:
        r_squared = 0.50
        rmse = 5.0
    
    # 6. CONSTRUIT LA RÉPONSE
    response = {
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
    
    # 7. AJOUTE L'EXPLICATION SHAP SI DISPONIBLE
    if explanation is not None:
        response['explanation'] = explanation
        print(f"   ✅ SHAP explanation included ({len(explanation.get('contributions', []))} features)")
    else:
        print(f"   ⚠️  No SHAP explanation available")
    
    return response


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
# ENDPOINTS PAR TYPE (POINTS, ASSISTS, REBOUNDS, 3PT) - RENDER PAYANT
# ============================================================================

@app.route('/api/daily-opportunities-points', methods=['GET'])
def daily_opportunities_points():
    """Scan 50 opportunités POINTS (RENDER PAYANT)"""
    return scan_opportunities_by_type('points', limit=50)


@app.route('/api/daily-opportunities-assists', methods=['GET'])
def daily_opportunities_assists():
    """Scan 50 opportunités ASSISTS (RENDER PAYANT)"""
    return scan_opportunities_by_type('assists', limit=50)


@app.route('/api/daily-opportunities-rebounds', methods=['GET'])
def daily_opportunities_rebounds():
    """Scan 50 opportunités REBOUNDS (RENDER PAYANT)"""
    return scan_opportunities_by_type('rebounds', limit=50)


@app.route('/api/daily-opportunities-3pt', methods=['GET'])
def daily_opportunities_3pt():
    """Scan 50 opportunités 3-POINTS (RENDER PAYANT)"""
    return scan_opportunities_by_type('3pt', limit=50)


def scan_opportunities_by_type(stat_type, limit=50):
    """
    Scan opportunités pour UN type de stat avec randomisation
    OPTIMISÉ POUR RENDER PAYANT (2 GB RAM, 120s timeout)
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
    print(f"🚀 SCANNING {stat_type.upper()} - {limit} PROPS (RENDER PAYANT)")
    print(f"{'='*70}\n")
    
    try:
        # Récupère TOUTES les props
        all_props = odds_client.get_player_props(days=1)
        
        # Map stat_type vers le type attendu par l'API
        stat_type_map = {
            'points': 'points',
            'assists': 'assists',
            'rebounds': 'rebounds',
            '3pt': '3pt'  # ← NOUVEAU
        }
        
        api_stat_type = stat_type_map.get(stat_type, stat_type)
        
        # Filtre par 'stat_type'
        filtered_props = [
            p for p in all_props 
            if p.get('stat_type') == api_stat_type
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
            'render_tier': 'PAID',  # ← NOUVEAU: Indicateur
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
        
        df = collector.get_complete_player_data(player_name)
        
        if df is None or len(df) == 0:
            return jsonify({
                'status': 'ERROR',
                'message': f'No data found for {player_name}'
            }), 404
        
        df = df.sort_values('GAME_DATE', ascending=False)
        recent_games = df.head(10)
        
        games = []
        for _, row in recent_games.iterrows():
            games.append({
                'date': row['GAME_DATE'],
                'opponent': row['MATCHUP'].split()[-1],
                'is_home': 'vs' in row['MATCHUP'],
                'points': int(row['PTS']),
                'assists': int(row['AST']),
                'rebounds': int(row['REB']),
                'three_pointers': int(row.get('FG3M', 0)),  # ← NOUVEAU
                'minutes': int(row['MIN']),
                'result': 'W' if row.get('WL', '') == 'W' else 'L'
            })
        
        pts_last_5 = df.head(5)['PTS'].mean()
        pts_prev_5 = df.iloc[5:10]['PTS'].mean() if len(df) >= 10 else df['PTS'].mean()
        pts_trend = pts_last_5 - pts_prev_5
        
        recent_wl = df.head(5)['WL'].value_counts().to_dict() if 'WL' in df.columns else {}
        wins = recent_wl.get('W', 0)
        losses = recent_wl.get('L', 0)
        
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
                'avg_3pt': round(df.get('FG3M', pd.Series([0])).mean(), 1),  # ← NOUVEAU
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
# AUTRES ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'OK',
        'xgboost_enabled': XGBOOST_AVAILABLE,
        'odds_api_enabled': ODDS_API_AVAILABLE,
        'render_tier': 'PAID',
        'max_props_per_scan': 50,
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
    print("NBA BETTING ANALYZER - API BACKEND (RENDER PAYANT)")
    print("="*70)
    print(f"Mode: {'XGBoost ✅' if XGBOOST_AVAILABLE else '❌'}")
    print(f"Odds API: {'✅' if ODDS_API_AVAILABLE else '❌'}")
    print(f"Render: PAID (50 props/scan)")
    print(f"Port: {port}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)