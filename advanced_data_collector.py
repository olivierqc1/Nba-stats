#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Data Collector - NBA API - PART 1/2
Récupère 35+ variables prédictives pour chaque joueur

⚠️ CE FICHIER EST INCOMPLET!
📋 INSTRUCTIONS:
1. Télécharge PART1 et PART2
2. Copie TOUT le contenu de PART1
3. Copie TOUT le contenu de PART2 EN DESSOUS
4. Sauvegarde comme "advanced_data_collector.py"
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import (
    playergamelog, 
    leaguedashteamstats,
    teamestimatedmetrics,
    playerdashboardbygeneralsplits,
    commonteamroster
)
from nba_api.stats.static import teams, players
import time

class AdvancedDataCollector:
    """
    Collecte données avancées pour améliorer les prédictions
    """
    
    def __init__(self):
        self.teams_dict = {team['abbreviation']: team['id'] for team in teams.get_teams()}
        self.cache = {}
        
    def get_complete_player_data(self, player_name, season='2024-25'):
        """
        Récupère TOUTES les données pour un joueur
        
        Returns:
            DataFrame avec 35+ colonnes de features
        """
        print(f"\n📥 Collecte données complètes pour {player_name}...")
        
        df_base = self._get_base_game_logs(player_name, season)
        
        if df_base is None or len(df_base) == 0:
            return None
        
        df_base = self._add_recent_form_features(df_base)
        df_base = self._add_usage_efficiency_features(df_base, player_name, season)
        df_base = self._add_opponent_features(df_base, season)
        df_base = self._add_team_context_features(df_base, season)
        df_base = self._add_temporal_features(df_base, season)
        df_base = self._add_advanced_splits(df_base, player_name)
        df_base = self._add_interaction_terms(df_base)
        df_base = self._add_polynomial_features(df_base)
        
        print(f"✅ {len(df_base.columns)} features collectées\n")
        
        return df_base
    
    def _get_base_game_logs(self, player_name, season):
        """Récupère les game logs de base"""
        try:
            player_id = self._get_player_id(player_name)
            
            if player_id is None:
                print(f"❌ Joueur non trouvé: {player_name}")
                return None
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            time.sleep(0.6)
            
            df = gamelog.get_data_frames()[0]
            
            df = df[[
                'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 
                'PTS', 'AST', 'REB', 'FGM', 'FGA', 
                'FG3M', 'FG3A', 'FTM', 'FTA', 'STL', 'BLK', 'TOV'
            ]].copy()
            
            df['is_home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            df['opponent'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
            
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            
            df['rest_days'] = df['GAME_DATE'].diff().dt.days.fillna(2)
            df['back_to_back'] = (df['rest_days'] == 1).astype(int)
            
            df['MIN'] = df['MIN'].apply(self._parse_minutes)
            
            print(f"  → {len(df)} matchs récupérés")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur game logs: {e}")
            return None
    
    def _parse_minutes(self, min_str):
        """Convertit '35:24' -> 35.4"""
        try:
            if pd.isna(min_str) or min_str == '' or min_str == 0:
                return 0.0
            if isinstance(min_str, (int, float)):
                return float(min_str)
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1])/60
        except:
            return 0.0
    
    def _get_player_id(self, player_name):
        """Trouve l'ID d'un joueur"""
        try:
            all_players = players.get_players()
            player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
            return player[0]['id'] if player else None
        except:
            return None
    
    def _add_recent_form_features(self, df):
        """Ajoute features de forme récente"""
        
        df['avg_last_3'] = df['PTS'].rolling(3, min_periods=1).mean().shift(1)
        df['avg_last_5'] = df['PTS'].rolling(5, min_periods=1).mean().shift(1)
        df['avg_last_10'] = df['PTS'].rolling(10, min_periods=1).mean().shift(1)
        
        df['std_last_5'] = df['PTS'].rolling(5, min_periods=1).std().shift(1)
        
        df['trend_5'] = self._calculate_rolling_trend(df['PTS'], window=5)
        
        df['std_last_5'] = df['std_last_5'].fillna(df['PTS'].std())
        
        return df
    
    def _calculate_rolling_trend(self, series, window=5):
        """Calcule la tendance sur fenêtre glissante"""
        trends = []
        for i in range(len(series)):
            if i < window:
                trends.append(0)
            else:
                x = np.arange(window)
                y = series.iloc[i-window:i].values
                if len(y) == window and not np.isnan(y).any():
                    slope = np.polyfit(x, y, 1)[0]
                    trends.append(slope)
                else:
                    trends.append(0)
        return pd.Series(trends, index=series.index)
    
    def _add_usage_efficiency_features(self, df, player_name, season):
        """Ajoute usage rate et efficacité"""
        
        df['usage_rate'] = ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN']) * 5
        df['usage_rate'] = df['usage_rate'].fillna(df['usage_rate'].mean())
        
        df['true_shooting_pct'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
        df['true_shooting_pct'] = df['true_shooting_pct'].fillna(0.55) * 100
        
        df['pts_per_shot'] = df['PTS'] / (df['FGA'] + 0.44 * df['FTA'])
        df['pts_per_shot'] = df['pts_per_shot'].fillna(1.0)
        
        df['efg_pct'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
        df['efg_pct'] = df['efg_pct'].fillna(0.5) * 100
        
        return df
    
    def _add_opponent_features(self, df, season):
        """Ajoute features sur l'adversaire"""
        
        team_def_stats = self._get_team_defensive_stats(season)
        
        df['opp_def_rating'] = df['opponent'].map(
            lambda x: team_def_stats.get(x, {}).get('def_rating', 110.0)
        )
        
        df['opp_def_rank'] = df['opponent'].map(
            lambda x: team_def_stats.get(x, {}).get('def_rank', 15)
        )
        
        df['opp_pace'] = df['opponent'].map(
            lambda x: team_def_stats.get(x, {}).get('pace', 99.0)
        )
        
        df['opp_pts_allowed'] = df['opponent'].map(
            lambda x: team_def_stats.get(x, {}).get('pts_allowed', 112.0)
        )
        
        df['historical_vs_opp'] = df.groupby('opponent')['PTS'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['historical_vs_opp'] = df['historical_vs_opp'].fillna(df['PTS'].mean())
        
        return df
    
    def _get_team_defensive_stats(self, season):
        """Récupère stats défensives de toutes les équipes"""
        
        cache_key = f"team_def_{season}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Advanced'
            )
            
            time.sleep(0.6)
            
            df = team_stats.get_data_frames()[0]
            
            stats_dict = {}
            
            for idx, row in df.iterrows():
                team_abbr = row['TEAM_ABBREVIATION']
                stats_dict[team_abbr] = {
                    'def_rating': row.get('DEF_RATING', 110.0),
                    'def_rank': idx + 1,
                    'pace': row.get('PACE', 99.0),
                    'pts_allowed': row.get('OPP_PTS', 112.0)
                }
            
            self.cache[cache_key] = stats_dict
            return stats_dict
            
        except Exception as e:
            print(f"  ⚠️  Erreur team stats: {e}, utilise valeurs par défaut")
            return {team: {
                'def_rating': 110.0,
                'def_rank': 15,
                'pace': 99.0,
                'pts_allowed': 112.0
            } for team in self.teams_dict.keys()}
    
    def _add_team_context_features(self, df, season):
        """Ajoute contexte de l'équipe"""
        
        df['teammates_out'] = 0
        df['key_players_out'] = 0
        df['team_pace'] = 99.5
        df['expected_possessions'] = df['team_pace'] * df['opp_pace'] / 100
        
        return df
    
    def _add_temporal_features(self, df, season):
        """Ajoute features temporelles"""
        
        df['season_progress'] = np.linspace(0, 1, len(df))
        df['is_playoff_race'] = (df['season_progress'] > 0.75).astype(int)
        df['days_since_injury'] = 999
        
        return df

# ============================================================================
# FIN DE PART 1/2
# ⚠️ CONTINUE AVEC advanced_data_collector_PART2.py
# ============================================================================
# ============================================================================
# ADVANCED_DATA_COLLECTOR.PY - PART 2/2
# ============================================================================
# 
# ⚠️ CE FICHIER COMMENCE AU MILIEU DU CODE!
# 
# 📋 INSTRUCTIONS:
# 1. Ouvre advanced_data_collector_PART1.py
# 2. Copie TOUT son contenu
# 3. Copie TOUT le contenu ci-dessous EN DESSOUS
# 4. Sauvegarde le tout comme "advanced_data_collector.py"
# 
# ============================================================================

    def _add_advanced_splits(self, df, player_name):
        """Ajoute splits home/away, vs good/bad teams"""
        
        home_avg = df[df['is_home'] == 1]['PTS'].mean()
        away_avg = df[df['is_home'] == 0]['PTS'].mean()
        df['home_away_diff'] = home_avg - away_avg
        
        df['vs_good_team'] = (df['opp_def_rank'] <= 15).astype(int)
        
        df['recent_minutes_avg'] = df['MIN'].rolling(5, min_periods=1).mean()
        
        return df
    
    def _add_interaction_terms(self, df):
        """Crée interactions entre variables"""
        
        df['home_x_def'] = df['is_home'] * df['opp_def_rank']
        df['rest_x_b2b'] = df['rest_days'] * df['back_to_back']
        df['usage_x_injuries'] = df['usage_rate'] * (df['teammates_out'] + 1)
        df['trend_x_importance'] = df['trend_5'] * df['is_playoff_race']
        df['form_x_matchup'] = df['avg_last_5'] * (31 - df['opp_def_rank']) / 30
        df['pace_interaction'] = df['team_pace'] * df['opp_pace'] / 100
        
        return df
    
    def _add_polynomial_features(self, df):
        """Ajoute features polynomiales"""
        
        df['minutes_squared'] = df['MIN'] ** 2
        df['rest_days_squared'] = df['rest_days'] ** 2
        
        return df
    
    def prepare_features_for_prediction(self, player_name, opponent, is_home, season='2024-25'):
        """
        Prépare le vecteur de features pour une prédiction
        
        Args:
            player_name: Nom du joueur
            opponent: Code équipe adverse (ex: 'GSW')
            is_home: True si domicile
            season: Saison
        
        Returns:
            dict avec toutes les features
        """
        
        df = self.get_complete_player_data(player_name, season)
        
        if df is None or len(df) == 0:
            return None
        
        last_game = df.iloc[-1]
        recent_5 = df.tail(5)
        recent_10 = df.tail(10)
        
        features = {
            # Base
            'is_home': 1 if is_home else 0,
            'MIN': recent_10['MIN'].mean(),
            'rest_days': 1,
            'back_to_back': 0,
            
            # Forme récente
            'avg_last_3': recent_5['PTS'].tail(3).mean(),
            'avg_last_5': recent_5['PTS'].mean(),
            'avg_last_10': recent_10['PTS'].mean(),
            'std_last_5': recent_5['PTS'].std(),
            'trend_5': self._calculate_rolling_trend(df['PTS'], window=5).iloc[-1],
            
            # Usage & Efficacité
            'usage_rate': recent_10['usage_rate'].mean(),
            'true_shooting_pct': recent_10['true_shooting_pct'].mean(),
            'pts_per_shot': recent_10['pts_per_shot'].mean(),
            'efg_pct': recent_10['efg_pct'].mean(),
            
            # Opponent
            'opp_def_rating': self._get_opponent_stat(opponent, 'def_rating', season),
            'opp_def_rank': self._get_opponent_stat(opponent, 'def_rank', season),
            'opp_pace': self._get_opponent_stat(opponent, 'pace', season),
            'opp_pts_allowed': self._get_opponent_stat(opponent, 'pts_allowed', season),
            'historical_vs_opp': df[df['opponent'] == opponent]['PTS'].mean() if len(df[df['opponent'] == opponent]) > 0 else df['PTS'].mean(),
            
            # Team context
            'teammates_out': 0,
            'key_players_out': 0,
            'team_pace': 99.5,
            'expected_possessions': 99.5 * self._get_opponent_stat(opponent, 'pace', season) / 100,
            
            # Temporel
            'season_progress': len(df) / 82,
            'is_playoff_race': int(len(df) / 82 > 0.75),
            'days_since_injury': 999,
            
            # Splits
            'home_away_diff': df[df['is_home'] == 1]['PTS'].mean() - df[df['is_home'] == 0]['PTS'].mean(),
            'vs_good_team': int(self._get_opponent_stat(opponent, 'def_rank', season) <= 15),
            'recent_minutes_avg': recent_5['MIN'].mean(),
        }
        
        # Interaction terms
        features['home_x_def'] = features['is_home'] * features['opp_def_rank']
        features['rest_x_b2b'] = features['rest_days'] * features['back_to_back']
        features['usage_x_injuries'] = features['usage_rate'] * (features['teammates_out'] + 1)
        features['trend_x_importance'] = features['trend_5'] * features['is_playoff_race']
        features['form_x_matchup'] = features['avg_last_5'] * (31 - features['opp_def_rank']) / 30
        features['pace_interaction'] = features['team_pace'] * features['opp_pace'] / 100
        
        # Polynomial
        features['minutes_squared'] = features['MIN'] ** 2
        features['rest_days_squared'] = features['rest_days'] ** 2
        
        return features
    
    def _get_opponent_stat(self, opponent, stat_name, season):
        """Récupère une stat de l'adversaire"""
        team_stats = self._get_team_defensive_stats(season)
        return team_stats.get(opponent, {}).get(stat_name, {
            'def_rating': 110.0,
            'def_rank': 15,
            'pace': 99.0,
            'pts_allowed': 112.0
        }[stat_name])


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    collector = AdvancedDataCollector()
    
    print("\n" + "="*70)
    print("TEST: Collecte données avancées")
    print("="*70)
    
    player = "LeBron James"
    df = collector.get_complete_player_data(player, '2024-25')
    
    if df is not None:
        print(f"\n✅ Données collectées:")
        print(f"  Matchs: {len(df)}")
        print(f"  Features: {len(df.columns)}")
        print(f"\n📊 Colonnes disponibles:")
        for col in sorted(df.columns):
            print(f"  - {col}")
        
        print(f"\n\n🎯 Test prédiction: {player} vs GSW (Domicile)")
        features = collector.prepare_features_for_prediction(player, 'GSW', True)
        
        if features:
            print(f"\n📋 Features prêtes pour prédiction:")
            print(f"  Total features: {len(features)}")
            print(f"\n  Top 5 features:")
            for i, (key, val) in enumerate(list(features.items())[:5]):
                print(f"    {key}: {val:.2f}")
    
    print("\n" + "="*70 + "\n")

# ============================================================================
# FIN DE PART 2/2
# ✅ FICHIER COMPLET!
# ============================================================================