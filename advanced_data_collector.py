#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Data Collector - VERSION SIMPLIFIÉE
Sans les features team qui causent des erreurs
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import teams, players
import time

class AdvancedDataCollector:
    """Collecte données pour améliorer prédictions - VERSION SIMPLIFIÉE"""
    
    def __init__(self):
        self.teams_dict = {team['abbreviation']: team['id'] for team in teams.get_teams()}
        self.cache = {}
        
    def get_complete_player_data(self, player_name, season='2024-25'):
        """Récupère données complètes pour un joueur"""
        print(f"\n📥 Collecte données pour {player_name}...")
        
        # 1. Game logs de base
        df = self._get_base_game_logs(player_name, season)
        
        if df is None or len(df) == 0:
            return None
        
        # 2. Features simples et fiables
        df = self._add_simple_features(df)
        
        print(f"✅ {len(df.columns)} features collectées\n")
        
        return df
    
    def _get_player_id(self, player_name):
        """Trouve l'ID d'un joueur"""
        all_players = players.get_players()
        player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
        return player[0]['id'] if player else None
    
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
            
            # Colonnes essentielles
            df = df[[
                'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 
                'PTS', 'AST', 'REB', 'FGM', 'FGA', 
                'FG3M', 'FG3A', 'FTM', 'FTA',
                'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
            ]].copy()
            
            # Parse date
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            # Minutes en float
            df['MIN'] = df['MIN'].fillna('0:00')
            df['MIN'] = df['MIN'].apply(self._parse_minutes)
            
            # Home/Away
            df['is_home'] = df['MATCHUP'].str.contains('vs.').astype(int)
            
            # Opponent
            df['opponent'] = df['MATCHUP'].apply(self._extract_opponent)
            
            # Win/Loss
            df['won'] = (df['WL'] == 'W').astype(int)
            
            print(f"   ✅ {len(df)} matchs récupérés")
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def _parse_minutes(self, min_str):
        """Convertit '34:25' en 34.42"""
        if pd.isna(min_str) or min_str == '0:00':
            return 0.0
        try:
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        except:
            return 0.0
    
    def _extract_opponent(self, matchup):
        """Extrait l'équipe adverse de 'LAL vs. BOS' → 'BOS'"""
        try:
            if 'vs.' in matchup:
                return matchup.split('vs.')[-1].strip()
            elif '@' in matchup:
                return matchup.split('@')[-1].strip()
            return 'UNK'
        except:
            return 'UNK'
    
    def _add_simple_features(self, df):
        """Ajoute features simples et fiables"""
        
        # MOYENNES RÉCENTES
        for window in [3, 5, 10]:
            df[f'avg_pts_last_{window}'] = df['PTS'].shift(1).rolling(window).mean()
            df[f'avg_ast_last_{window}'] = df['AST'].shift(1).rolling(window).mean()
            df[f'avg_reb_last_{window}'] = df['REB'].shift(1).rolling(window).mean()
            df[f'avg_min_last_{window}'] = df['MIN'].shift(1).rolling(window).mean()
        
        # TENDANCES
        df['trend_pts_5'] = df['PTS'].shift(1).rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        
        # STABILITÉ
        df['std_pts_10'] = df['PTS'].shift(1).rolling(10).std()
        df['std_min_10'] = df['MIN'].shift(1).rolling(10).std()
        
        # PERFORMANCE
        df['fg_pct'] = (df['FGM'] / df['FGA'] * 100).fillna(50.0)
        df['fg3_pct'] = (df['FG3M'] / df['FG3A'] * 100).fillna(35.0)
        df['ft_pct'] = (df['FTM'] / df['FTA'] * 100).fillna(75.0)
        
        # TRUE SHOOTING
        df['ts_pct'] = (df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])) * 100).fillna(55.0)
        
        # EFFICACITÉ
        df['pts_per_min'] = (df['PTS'] / df['MIN']).fillna(0.8)
        df['usage'] = ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN']).fillna(0.3)
        
        # FORME RÉCENTE (W/L)
        df['recent_wins_5'] = df['won'].shift(1).rolling(5).sum()
        df['recent_win_pct'] = df['recent_wins_5'] / 5
        
        # REST DAYS
        df['rest_days'] = df['GAME_DATE'].diff().dt.days.abs()
        df['rest_days'] = df['rest_days'].fillna(2.0).clip(0, 7)
        df['back_to_back'] = (df['rest_days'] == 1).astype(int)
        
        # HOME/AWAY SPLIT
        df['home_avg_pts'] = df[df['is_home'] == 1].groupby('is_home')['PTS'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['away_avg_pts'] = df[df['is_home'] == 0].groupby('is_home')['PTS'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['home_avg_pts'] = df['home_avg_pts'].fillna(df['PTS'].mean())
        df['away_avg_pts'] = df['away_avg_pts'].fillna(df['PTS'].mean())
        
        # VS OPPONENT HISTORIQUE
        df['vs_opp_avg_pts'] = df.groupby('opponent')['PTS'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        df['vs_opp_avg_pts'] = df['vs_opp_avg_pts'].fillna(df['PTS'].mean())
        
        # GAME NUMBER DANS SAISON
        df['games_played'] = range(len(df), 0, -1)
        df['season_fatigue'] = df['games_played'] / 82  # Normalize
        
        # MINUTES TREND
        df['min_trend_5'] = df['MIN'].shift(1).rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        )
        
        # CONSISTENCY
        df['pts_consistency'] = 1 / (1 + df['std_pts_10'].fillna(5.0))
        
        # INTERACTION TERMS
        df['min_x_fg_pct'] = df['MIN'] * df['fg_pct'] / 100
        df['rest_x_usage'] = df['rest_days'] * df['usage']
        df['home_x_recent_win'] = df['is_home'] * df['recent_win_pct']
        
        # POLYNOMIAL
        df['avg_pts_last_5_squared'] = df['avg_pts_last_5'] ** 2
        df['min_squared'] = df['MIN'] ** 2
        
        # FILL NAs
        df = df.fillna(df.mean())
        
        return df
    
    def prepare_features_for_prediction(self, player_name, opponent='', is_home=True, current_features=None):
        """Prépare features pour prédiction"""
        
        # Si features déjà fournis
        if current_features is not None and isinstance(current_features, pd.DataFrame):
            return current_features
        
        # Sinon récupère données complètes
        df = self.get_complete_player_data(player_name)
        
        if df is None or len(df) == 0:
            return None
        
        # Prend les features du match le plus récent (ligne 0)
        features = df.iloc[0:1].copy()
        
        # Drop colonnes non-numériques
        features = features.select_dtypes(include=[np.number])
        
        # Drop colonnes cibles
        features = features.drop(columns=['PTS', 'AST', 'REB'], errors='ignore')
        
        return features


if __name__ == '__main__':
    # Test rapide
    collector = AdvancedDataCollector()
    df = collector.get_complete_player_data('LeBron James')
    
    if df is not None:
        print(f"\n✅ SUCCESS: {len(df)} matchs, {len(df.columns)} features")
        print(f"Features: {df.columns.tolist()}")
    else:
        print("\n❌ FAILED")