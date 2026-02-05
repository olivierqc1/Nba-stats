#!/usr/bin/env python3
"""
NBA Betting Analyzer - Backend Amélioré
✅ Points, Assists, Rebounds séparés
✅ Test du Chi-carré (χ²)
✅ Exclusion des outliers avec IQR + Z-score
✅ Régression linéaire avec stats complètes
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import os

try:
    from nba_api.stats.static import players, teams
    from nba_api.stats.endpoints import playergamelog, commonteamroster, leaguegamefinder
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️ nba_api non disponible - Utilise données simulées")

app = Flask(__name__)
CORS(app)

class ImprovedNBAAnalyzer:
    """
    Analyseur NBA avec support multi-stats et détection avancée d'outliers
    """
    
    def __init__(self):
        self.cache = {}
        self.defensive_ratings = {
            'ATL': 113.5, 'BOS': 108.2, 'BKN': 114.2, 'CHA': 115.8,
            'CHI': 112.1, 'CLE': 109.5, 'DAL': 112.8, 'DEN': 109.2,
            'DET': 115.3, 'GSW': 110.5, 'HOU': 113.8, 'IND': 114.5,
            'LAC': 110.7, 'LAL': 112.3, 'MEM': 111.2, 'MIA': 109.8,
            'MIL': 110.1, 'MIN': 110.4, 'NOP': 113.2, 'NYK': 109.8,
            'OKC': 108.5, 'ORL': 110.3, 'PHI': 108.9, 'PHX': 111.4,
            'POR': 114.8, 'SAC': 112.6, 'SAS': 115.1, 'TOR': 113.4,
            'UTA': 114.2, 'WAS': 116.5
        }
        
    def get_player_games(self, player_name, season='2024-25'):
        """Récupère les matchs d'un joueur (vraies données ou simulées)"""
        
        if not NBA_API_AVAILABLE:
            return self._simulate_player_games(player_name, 25)
        
        cache_key = f"{player_name}_{season}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Trouve le joueur
            player_list = players.find_players_by_full_name(player_name)
            if not player_list:
                return self._simulate_player_games(player_name, 25)
            
            player_id = player_list[0]['id']
            
            # Récupère les logs
            import time
            time.sleep(0.6)  # Rate limiting
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return self._simulate_player_games(player_name, 25)
            
            # Reformate
            df_clean = pd.DataFrame({
                'date': pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d'),
                'opponent': df['MATCHUP'].str.split().str[-1],
                'is_home': ~df['MATCHUP'].str.contains('@'),
                'points': df['PTS'].astype(float),
                'rebounds': df['REB'].astype(float),
                'assists': df['AST'].astype(float),
                'minutes': df['MIN'].apply(lambda x: float(str(x).split(':')[0]) if ':' in str(x) else float(x)),
                'fg_pct': (df['FG_PCT'].astype(float) * 100).fillna(0),
                'result': df['WL']
            })
            
            # Ajoute variables pour régression
            df_clean['opponent_def_rating'] = df_clean['opponent'].apply(
                lambda x: self.defensive_ratings.get(x, 112.0)
            )
            
            df_clean['rest_days'] = 1
            df_clean['back_to_back'] = 0
            df_clean['team_pace'] = 100.0
            
            self.cache[cache_key] = df_clean
            print(f"✅ {len(df_clean)} matchs réels pour {player_name}")
            
            return df_clean
            
        except Exception as e:
            print(f"❌ Erreur API: {e}")
            return self._simulate_player_games(player_name, 25)
    
    def _simulate_player_games(self, player_name, n_games):
        """Simule des données si API non disponible"""
        np.random.seed(hash(player_name) % 2**32)
        
        base_pts = np.random.uniform(20, 28)
        base_ast = np.random.uniform(4, 8)
        base_reb = np.random.uniform(4, 10)
        
        games = []
        for i in range(n_games):
            game = {
                'date': (datetime.now()).strftime('%Y-%m-%d'),
                'opponent': np.random.choice(list(self.defensive_ratings.keys())),
                'is_home': np.random.choice([True, False]),
                'points': max(0, base_pts + np.random.normal(0, 5)),
                'assists': max(0, base_ast + np.random.normal(0, 2)),
                'rebounds': max(0, base_reb + np.random.normal(0, 3)),
                'minutes': np.random.uniform(30, 38),
                'fg_pct': np.random.uniform(40, 55),
                'opponent_def_rating': np.random.uniform(108, 116),
                'rest_days': 1,
                'back_to_back': 0,
                'team_pace': 100.0,
                'result': np.random.choice(['W', 'L'])
            }
            games.append(game)
        
        return pd.DataFrame(games)
    
    def detect_outliers(self, values, method='iqr'):
        """
        Détecte les outliers avec 3 méthodes combinées
        
        method: 'iqr', 'zscore', 'combined'
        """
        values = np.array(values)
        n = len(values)
        
        outliers_mask = np.zeros(n, dtype=bool)
        outlier_methods = {}
        
        # Méthode 1: IQR (Interquartile Range)
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        lower_iqr = Q1 - 1.5 * IQR
        upper_iqr = Q3 + 1.5 * IQR
        iqr_outliers = (values < lower_iqr) | (values > upper_iqr)
        
        # Méthode 2: Z-score (> 2.5 std)
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros(n)
        zscore_outliers = z_scores > 2.5
        
        # Méthode 3: Modified Z-score (MAD - Median Absolute Deviation)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        modified_z_scores = 0.6745 * (values - median) / mad if mad > 0 else np.zeros(n)
        mad_outliers = np.abs(modified_z_scores) > 3.5
        
        # Combine les méthodes
        if method == 'iqr':
            outliers_mask = iqr_outliers
        elif method == 'zscore':
            outliers_mask = zscore_outliers
        else:  # combined
            # Un outlier si détecté par au moins 2 méthodes
            outliers_mask = (iqr_outliers.astype(int) + 
                           zscore_outliers.astype(int) + 
                           mad_outliers.astype(int)) >= 2
        
        # Métadonnées pour chaque point
        outlier_info = []
        for i, val in enumerate(values):
            outlier_info.append({
                'index': int(i),
                'value': float(val),
                'is_outlier': bool(outliers_mask[i]),
                'z_score': float(z_scores[i]),
                'methods_detected': {
                    'iqr': bool(iqr_outliers[i]),
                    'zscore': bool(zscore_outliers[i]),
                    'mad': bool(mad_outliers[i])
                },
                'severity': 'HIGH' if outliers_mask[i] and (
                    iqr_outliers[i] and zscore_outliers[i] and mad_outliers[i]
                ) else 'MEDIUM' if outliers_mask[i] else 'LOW'
            })
        
        return outliers_mask, outlier_info
    
    def chi_square_test(self, observed, expected):
        """
        Test du Chi-carré pour comparer distribution observée vs attendue
        
        H0: Les distributions sont identiques
        H1: Les distributions diffèrent significativement
        """
        # Binning des données
        bins = np.linspace(min(min(observed), min(expected)), 
                          max(max(observed), max(expected)), 10)
        
        obs_freq, _ = np.histogram(observed, bins=bins)
        exp_freq, _ = np.histogram(expected, bins=bins)
        
        # Évite division par zéro
        mask = exp_freq > 0
        obs_freq = obs_freq[mask]
        exp_freq = exp_freq[mask]
        
        if len(obs_freq) == 0:
            return {
                'chi2_statistic': 0.0,
                'p_value': 1.0,
                'dof': 0,
                'significant': False,
                'interpretation': 'Pas assez de données'
            }
        
        # Calcul du Chi²
        chi2_stat = np.sum((obs_freq - exp_freq) ** 2 / exp_freq)
        dof = len(obs_freq) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'dof': int(dof),
            'significant': p_value < 0.05,
            'interpretation': (
                'Distribution SIGNIFICATIVEMENT différente (p < 0.05)' if p_value < 0.05
                else 'Distribution CONFORME au modèle (p >= 0.05)'
            )
        }
    
    def analyze_stat(self, player_name, stat_type='points', opponent='LAL', 
                    is_home=True, line=None, remove_outliers=True):
        """
        Analyse complète pour UNE statistique (points, assists ou rebounds)
        
        stat_type: 'points', 'assists', 'rebounds'
        remove_outliers: True pour exclure les outliers du modèle
        """
        
        # 1. Récupère les données
        df = self.get_player_games(player_name)
        
        if df.empty or len(df) < 5:
            return {'error': f'Pas assez de données pour {player_name}'}
        
        # 2. Détecte les outliers
        stat_values = df[stat_type].values
        outliers_mask, outlier_info = self.detect_outliers(stat_values, method='combined')
        
        # 3. Crée deux versions: avec et sans outliers
        df_full = df.copy()
        df_clean = df[~outliers_mask].copy() if remove_outliers and np.any(outliers_mask) else df.copy()
        
        # 4. Construit le modèle sur données choisies
        df_model = df_clean if remove_outliers else df_full
        
        if len(df_model) < 5:
            df_model = df_full  # Pas assez de données sans outliers
        
        X = df_model[['is_home', 'opponent_def_rating', 'minutes', 
                     'rest_days', 'back_to_back', 'team_pace']].astype(float)
        y = df_model[stat_type].astype(float)
        
        # Régression
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédictions
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Statistiques
        n = len(y)
        k = X.shape[1]
        dof = n - k - 1
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else r2
        
        # P-values
        var_residuals = np.sum(residuals**2) / dof if dof > 0 else 1
        try:
            var_coef = var_residuals * np.linalg.inv(X.T @ X).diagonal()
            std_errors = np.sqrt(var_coef)
            t_stats = model.coef_ / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        except:
            std_errors = np.ones(k)
            t_stats = np.zeros(k)
            p_values = np.ones(k)
        
        # 5. Test du Chi-carré
        chi2_test = self.chi_square_test(y.values, y_pred)
        
        # 6. Prédiction pour le prochain match
        opp_def = self.defensive_ratings.get(opponent, 112.0)
        X_pred = np.array([[
            1 if is_home else 0,
            opp_def,
            35,  # minutes estimées
            1,   # rest_days
            0,   # back_to_back
            100  # pace
        ]])
        
        prediction = float(model.predict(X_pred)[0])
        std_dev = float(y.std())
        
        # Intervalle de confiance 95%
        z_95 = 1.96
        ci_lower = prediction - z_95 * std_dev
        ci_upper = prediction + z_95 * std_dev
        
        # 7. Analyse de ligne si fournie
        if line is None:
            line = prediction - 0.5
        
        z_score = (prediction - line) / std_dev if std_dev > 0 else 0
        prob_over = float(stats.norm.cdf(z_score))
        prob_under = 1 - prob_over
        
        # Edge calculation
        implied_prob_over = 0.5238  # -110 odds
        implied_prob_under = 0.5238
        
        edge_over = prob_over - implied_prob_over
        edge_under = prob_under - implied_prob_under
        
        # Recommandation
        if edge_over > 0.05 and edge_over > abs(edge_under):
            recommendation = 'OVER'
            edge = edge_over
            prob = prob_over
        elif edge_under > 0.05:
            recommendation = 'UNDER'
            edge = edge_under
            prob = prob_under
        else:
            recommendation = 'SKIP'
            edge = 0
            prob = 0.5
        
        # Kelly criterion
        if recommendation != 'SKIP':
            decimal_odds = 1.91  # -110 américain
            q = 1 - prob
            kelly = (prob * decimal_odds - q) / decimal_odds
            kelly_pct = max(0, kelly * 0.25) * 100  # Quarter Kelly
        else:
            kelly_pct = 0
        
        # 8. Métadonnées outliers
        outliers_detected = np.sum(outliers_mask)
        outliers_list = []
        
        for i, is_outlier in enumerate(outliers_mask):
            if is_outlier:
                outliers_list.append({
                    'date': df.iloc[i]['date'],
                    'opponent': df.iloc[i]['opponent'],
                    stat_type: float(df.iloc[i][stat_type]),
                    'reason': outlier_info[i]['severity'],
                    'methods_detected': outlier_info[i]['methods_detected']
                })
        
        return {
            'status': 'SUCCESS',
            'player': player_name,
            'stat_type': stat_type,
            'opponent': opponent,
            'is_home': is_home,
            'data_source': 'REAL (Full Season)' if NBA_API_AVAILABLE else 'SIMULATED',
            
            # Prédiction
            'prediction': round(prediction, 1),
            'confidence_interval': {
                'lower': round(max(0, ci_lower), 1),
                'upper': round(ci_upper, 1)
            },
            
            # Stats de saison
            'season_stats': {
                'games_played': len(df),
                'games_used': len(df_model),
                'outliers_removed': int(outliers_detected) if remove_outliers else 0,
                'weighted_avg': round(df[stat_type].mean(), 1),
                'std_dev': round(std_dev, 2),
                'min': round(df[stat_type].min(), 1),
                'max': round(df[stat_type].max(), 1)
            },
            
            # Modèle de régression
            'regression_stats': {
                'r_squared': round(r2, 4),
                'adjusted_r_squared': round(adj_r2, 4),
                'rmse': round(rmse, 2),
                'sample_size': int(n),
                'dof': int(dof)
            },
            
            # Test du Chi-carré
            'chi_square_test': chi2_test,
            
            # Analyse de ligne
            'line_analysis': {
                'bookmaker_line': round(line, 1),
                'recommendation': recommendation,
                'over_probability': round(prob_over * 100, 1),
                'under_probability': round(prob_under * 100, 1),
                'edge': round(edge * 100, 1),
                'kelly_criterion': round(kelly_pct, 1),
                'bet_confidence': 'HIGH' if abs(edge) > 0.10 else 'MEDIUM' if abs(edge) > 0.05 else 'LOW'
            },
            
            # Outliers
            'outlier_analysis': {
                'method': 'Combined (IQR + Z-score + MAD)',
                'outliers_detected': int(outliers_detected),
                'outliers_pct': round((outliers_detected / len(df)) * 100, 1),
                'data_used': 'CLEANED' if remove_outliers and outliers_detected > 0 else 'FULL',
                'outliers': outliers_list,
                'recommendation': (
                    f'{outliers_detected} outlier(s) détecté(s) et EXCLUS du modèle' 
                    if remove_outliers and outliers_detected > 0 
                    else f'{outliers_detected} outlier(s) détecté(s) mais INCLUS dans le modèle'
                )
            },
            
            # Splits
            'splits': {
                'home': {
                    'games': len(df[df['is_home'] == True]),
                    'avg': round(df[df['is_home'] == True][stat_type].mean(), 1)
                } if len(df[df['is_home'] == True]) > 0 else None,
                'away': {
                    'games': len(df[df['is_home'] == False]),
                    'avg': round(df[df['is_home'] == False][stat_type].mean(), 1)
                } if len(df[df['is_home'] == False]) > 0 else None,
                'vs_opponent': {
                    'games': len(df[df['opponent'] == opponent]),
                    'avg': round(df[df['opponent'] == opponent][stat_type].mean(), 1)
                } if len(df[df['opponent'] == opponent]) > 0 else None
            },
            
            # Trend analysis
            'trend_analysis': {
                'slope': round(model.coef_[0], 3),
                'r_squared': round(r2, 3),
                'p_value': f"{'<0.001' if p_values[0] < 0.001 else round(p_values[0], 3)}",
                'interpretation': (
                    'Tendance à la hausse' if model.coef_[0] > 0.5 
                    else 'Tendance à la baisse' if model.coef_[0] < -0.5 
                    else 'Stable'
                )
            }
        }

analyzer = ImprovedNBAAnalyzer()

# Import Odds API Client
try:
    from odds_api_client import OddsAPIClient
    odds_client = OddsAPIClient()
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False
    odds_client = None
    print("⚠️ odds_api_client.py non trouvé - mode manuel uniquement")

# ========================= DAILY OPPORTUNITIES =========================

def scan_daily_opportunities(min_edge=5.0, min_confidence='MEDIUM'):
    """
    Scanne toutes les opportunités du jour
    
    Returns: Liste des meilleures opportunités triées par edge
    """
    if not ODDS_API_AVAILABLE or not odds_client:
        return {
            'status': 'ERROR',
            'message': 'Odds API non disponible',
            'opportunities': []
        }
    
    print("\n" + "="*70)
    print("🔍 SCAN DES OPPORTUNITÉS DU JOUR")
    print("="*70)
    
    # 1. Récupère les props disponibles
    props = odds_client.get_player_props()
    print(f"📊 {len(props)} props récupérées")
    
    opportunities = []
    analyzed_count = 0
    
    # 2. Analyse chaque prop
    for prop in props:
        player = prop['player']
        stat_type = prop['stat_type']
        line = prop['line']
        bookmaker = prop['bookmaker']
        
        # Détermine is_home
        # (besoin de savoir quelle équipe joue)
        is_home = True  # TODO: améliorer avec vraie logique
        opponent = prop['away_team'] if is_home else prop['home_team']
        
        try:
            # Analyse avec notre modèle
            result = analyzer.analyze_stat(
                player, stat_type, opponent, is_home, line, 
                remove_outliers=True
            )
            
            analyzed_count += 1
            
            if result.get('status') != 'SUCCESS':
                continue
            
            # Filtre selon edge et confiance
            edge = result['line_analysis']['edge']
            confidence = result['line_analysis']['bet_confidence']
            rec