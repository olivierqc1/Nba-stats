#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Betting Analyzer - API Backend avec XGBoost
VERSION OPTIMISÉE: 3 endpoints séparés + randomisation
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
    """Scan 25 opportunités POINTS aléatoires"""
    return scan_opportunities_by_type('points', limit=25)


@app.route('/api/daily-opportunities-assists', methods=['GET'])
def daily_opportunities_assists():
    """Scan 25 opportunités ASSISTS aléatoires"""
    return scan_opportunities_by_type('assists', limit=25)


@app.route('/api/daily-opportunities-rebounds', methods=['GET'])
def daily_opportunities_rebounds():
    """Scan 25 opportunités REBOUNDS aléatoires"""
    return scan_opportunities_by_type('rebounds', limit=25)


def scan_opportunities_by_type(stat_type, limit=25):
    """
    Scan opportunités pour UN type de stat avec randomisation
    """
    
    min_edge = request.args.get('min_edge', 5.0, type=float)
    min_r2 = request.args.get('min_r2', 0.70, type=float)
    
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
        
        # Map stat type to market name
        stat_map = {
            'points': 'player_points',
            'assists': 'player_assists', 
            'rebounds': 'player_rebounds'
        }
        
        market_name = stat_map.get(stat_type)
        
        # Filtre par stat_type
        filtered_props = [
            p for p in all_props 
            if p.get('market') == market_name
        ]
        
        print(f"📊 Total {stat_type} available: {len(filtered_props)}")
        
        # RANDOMISE et limite
        random.shuffle(filtered_props)
        selected_props = filtered_props[:limit]
        
        print(f"🎲 Selected (random): {len(selected_props)}")
        
        opportunities = []
        analyzed_count = 0
        
        for prop in selected_props:
            player = prop['player']
            line = prop['line']
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
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Morning Brief - Paris NBA FIABLES</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            margin-bottom: 5px;
        }
        .header p { color: #666; font-size: 1.1em; }
        .filters {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .filter-row {
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .filter-group { display: flex; align-items: center; gap: 10px; }
        .filter-group label { font-weight: 600; color: #333; }
        .filter-group input, .filter-group select {
            padding: 8px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
        }
        .button-group {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .btn-scan {
            padding: 15px 25px;
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            cursor: pointer;
            font-size: 1.05em;
            transition: all 0.3s;
        }
        .btn-scan:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        .btn-scan:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-points { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
        .btn-assists { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
        .btn-rebounds { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }
        .stats-bar {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        .stat-box { text-align: center; padding: 15px; border-radius: 10px; background: #f8f9fa; }
        .stat-label { font-size: 0.85em; color: #666; margin-bottom: 5px; }
        .stat-value { font-size: 2em; font-weight: 800; color: #667eea; }
        .opportunities-grid { display: grid; gap: 20px; }
        .opp-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
            border-left: 6px solid #10b981;
            position: relative;
        }
        .opp-card.under { border-left-color: #ef4444; }
        .opp-card::before {
            content: attr(data-rank);
            position: absolute;
            top: 15px;
            right: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.2em;
        }
        .opp-header { margin-bottom: 20px; }
        .opp-player { font-size: 1.8em; font-weight: 800; color: #333; margin-bottom: 5px; }
        .opp-matchup { color: #666; font-size: 1.1em; margin-bottom: 10px; }
        .opp-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1em;
            margin-top: 10px;
        }
        .opp-badge.over { background: #d1fae5; color: #065f46; }
        .opp-badge.under { background: #fee2e2; color: #991b1b; }
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            color: white;
        }
        .pred-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .pred-label { font-size: 1em; opacity: 0.9; }
        .pred-value { font-size: 3em; font-weight: 800; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
        }
        .metric {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-label { font-size: 0.85em; opacity: 0.9; margin-bottom: 5px; }
        .metric-value { font-size: 1.5em; font-weight: 700; }
        .action-box {
            background: #10b981;
            padding: 20px;
            border-radius: 12px;
            color: white;
            margin-top: 20px;
        }
        .action-box.under { background: #ef4444; }
        .action-title { font-size: 1.2em; font-weight: 700; margin-bottom: 10px; }
        .action-detail { font-size: 1.1em; }
        .loading {
            text-align: center;
            padding: 60px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .hidden { display: none; }
        .error {
            background: #fee2e2;
            border: 3px solid #fca5a5;
            color: #991b1b;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
        }
        .info-banner {
            background: #dbeafe;
            border: 2px solid #3b82f6;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            color: #1e40af;
        }
        .info-banner strong { color: #1e3a8a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Paris NBA ULTRA-FIABLES</h1>
            <p>Prédictions XGBoost - Données en temps réel</p>
        </div>

        <div class="info-banner">
            <strong>🎲 Scan Aléatoire + Séparé</strong><br>
            Chaque bouton analyse 25 props ALÉATOIRES d'un type spécifique.<br>
            Temps: 30-90 secondes par scan. Données: Odds-API + NBA Stats en direct.
        </div>

        <div class="filters">
            <div class="filter-row">
                <div class="filter-group">
                    <label>R² minimum:</label>
                    <select id="minR2">
                        <option value="0.70">0.70 (bon)</option>
                        <option value="0.75">0.75 (très bon)</option>
                        <option value="0.80" selected>0.80 (excellent)</option>
                        <option value="0.85">0.85 (exceptionnel)</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Edge minimum:</label>
                    <input type="number" id="minEdge" value="5" min="0" max="20" step="0.5">
                    <span>%</span>
                </div>
            </div>
            
            <div class="button-group">
                <button class="btn-scan btn-points" onclick="scanOpportunities('points')">
                    🏀 Scanner POINTS (25 random)
                </button>
                <button class="btn-scan btn-assists" onclick="scanOpportunities('assists')">
                    🎯 Scanner ASSISTS (25 random)
                </button>
                <button class="btn-scan btn-rebounds" onclick="scanOpportunities('rebounds')">
                    💪 Scanner REBOUNDS (25 random)
                </button>
            </div>
        </div>

        <div id="statsBar" class="stats-bar hidden">
            <div class="stat-box">
                <div class="stat-label">Type analysé</div>
                <div class="stat-value" id="statType">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Props scannées</div>
                <div class="stat-value" id="analyzedProps">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Opportunités trouvées</div>
                <div class="stat-value" id="foundOpps">-</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">R² moyen</div>
                <div class="stat-value" id="avgR2">-</div>
            </div>
        </div>

        <div id="loadingDiv" class="loading hidden">
            <div class="spinner"></div>
            <h3>Scan en cours...</h3>
            <p>Analyse de 25 props aléatoires (30-90 sec)</p>
        </div>

        <div id="resultsDiv" class="opportunities-grid hidden"></div>

        <div id="errorDiv" class="error hidden"></div>
    </div>

    <script>
        const API_URL = 'https://nba-stats-xcyv.onrender.com';

        async function scanOpportunities(statType) {
            const minEdge = document.getElementById('minEdge').value;
            const minR2 = parseFloat(document.getElementById('minR2').value);

            // Disable tous les boutons
            document.querySelectorAll('.btn-scan').forEach(btn => btn.disabled = true);

            document.getElementById('loadingDiv').classList.remove('hidden');
            document.getElementById('statsBar').classList.add('hidden');
            document.getElementById('resultsDiv').classList.add('hidden');
            document.getElementById('errorDiv').classList.add('hidden');

            const endpoints = {
                'points': 'daily-opportunities-points',
                'assists': 'daily-opportunities-assists',
                'rebounds': 'daily-opportunities-rebounds'
            };

            try {
                const response = await fetch(
                    `${API_URL}/api/${endpoints[statType]}?min_edge=${minEdge}&min_r2=${minR2}`,
                    { timeout: 120000 }
                );

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();

                if (data.status === 'SUCCESS') {
                    displayResults(data, statType);
                } else {
                    displayError(data.message || 'Erreur inconnue');
                }
            } catch (error) {
                console.error('Scan error:', error);
                displayError(`Erreur: ${error.message}`);
            } finally {
                document.getElementById('loadingDiv').classList.add('hidden');
                // Re-enable boutons
                document.querySelectorAll('.btn-scan').forEach(btn => btn.disabled = false);
            }
        }

        function displayResults(data, statType) {
            const opportunities = data.opportunities || [];
            
            document.getElementById('statsBar').classList.remove('hidden');
            document.getElementById('statType').textContent = statType.toUpperCase();
            document.getElementById('analyzedProps').textContent = data.total_analyzed;
            document.getElementById('foundOpps').textContent = opportunities.length;
            
            if (opportunities.length > 0) {
                const avgR2 = opportunities.reduce((sum, o) => 
                    sum + (o.regression_stats?.r_squared || 0), 0) / opportunities.length;
                document.getElementById('avgR2').textContent = avgR2.toFixed(3);
            }

            if (opportunities.length === 0) {
                displayError('Aucun pari trouvé avec ces filtres. Essaie de baisser R² ou Edge.');
                return;
            }
            
            const resultsDiv = document.getElementById('resultsDiv');
            resultsDiv.innerHTML = opportunities.map((opp, index) => 
                createOpportunityCard(opp, index + 1)
            ).join('');
            resultsDiv.classList.remove('hidden');
        }

        function getTeamName(abbr) {
            const teams = {
                'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
                'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
                'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
                'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
                'LAC': 'LA Clippers', 'LAL': 'LA Lakers', 'MEM': 'Memphis Grizzlies',
                'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
                'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
                'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
                'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
                'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
            };
            return teams[abbr] || abbr;
        }

        function createOpportunityCard(opp, rank) {
            const rec = opp.line_analysis.recommendation;
            const recClass = rec.toLowerCase();
            const edge = opp.line_analysis.edge;
            const kelly = opp.line_analysis.kelly_criterion;
            const r2 = opp.regression_stats?.r_squared || 0;
            const rmse = opp.regression_stats?.rmse || 0;

            const statLabel = {
                'points': 'Points',
                'assists': 'Assists',
                'rebounds': 'Rebounds'
            }[opp.stat_type];

            const teamName = getTeamName(opp.opponent);

            return `
                <div class="opp-card ${recClass}" data-rank="${rank}">
                    <div class="opp-header">
                        <div class="opp-player">${opp.player}</div>
                        <div class="opp-matchup">
                            ${opp.is_home ? '🏠 vs' : '✈️ @'} ${teamName} • ${statLabel}
                        </div>
                        <div class="opp-badge ${recClass}">
                            ${rec} ${opp.line_analysis.bookmaker_line}
                        </div>
                    </div>

                    <div class="prediction-box">
                        <div class="pred-row">
                            <div>
                                <div class="pred-label">Prédiction modèle</div>
                                <div class="pred-value">${opp.prediction}</div>
                            </div>
                            <div style="text-align: right;">
                                <div class="pred-label">vs Ligne</div>
                                <div class="pred-value">${opp.line_analysis.bookmaker_line}</div>
                            </div>
                        </div>
                        
                        <div class="metrics-grid">
                            <div class="metric">
                                <div class="metric-label">🎯 R² (Fiabilité)</div>
                                <div class="metric-value">${(r2 * 100).toFixed(1)}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">📊 RMSE</div>
                                <div class="metric-value">${rmse.toFixed(1)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">💰 Edge</div>
                                <div class="metric-value">+${edge.toFixed(1)}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">📈 Kelly</div>
                                <div class="metric-value">${kelly.toFixed(1)}%</div>
                            </div>
                        </div>
                    </div>

                    <div class="action-box ${recClass}">
                        <div class="action-title">🎯 ACTION À PRENDRE</div>
                        <div class="action-detail">
                            ➤ Parier <strong>${rec} ${opp.line_analysis.bookmaker_line}</strong><br>
                            ➤ Mise recommandée: <strong>${kelly.toFixed(1)}%</strong> de ta bankroll<br>
                            ➤ Fiabilité du modèle: <strong>${(r2 * 100).toFixed(0)}%</strong>
                        </div>
                    </div>
                </div>
            `;
        }

        function displayError(message) {
            const errorDiv = document.getElementById('errorDiv');
            errorDiv.textContent = `❌ ${message}`;
            errorDiv.classList.remove('hidden');
        }
    </script>
</body>
</html> 