#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Data Collector - VERSION ULTRA-ROBUSTE
Ne crash JAMAIS - Try/except partout
"""

import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import teams, players
import time

class AdvancedDataCollector:
    """Collecte données - VERSION INCASSABLE"""
    
    def __init__(self):
        try:
            self.teams_dict = {team['abbreviation']: team['id'] for team in teams.get_teams()}
        except:
            self.teams_dict = {}
        self.cache = {}
        
    def get_complete_player_data(self, player_name, season='2024-25'):
        """Récupère données - NE CRASH JAMAIS"""
        try:
            print(f"\n📥 {player_name}...")
            
            df = self._get_base_game_logs(player_name, season)
            
            if df is None or len(df) == 0:
                print(f"❌ No data for {player_name}")
                return None
            
            df = self._add_simple_features(df)
            
            print(f"✅ {len(df)} games, {len(df.columns)} features")
            
            return df
        except Exception as e:
            print(f"❌ CRASH in get_complete_player_data: {e}")
            return None
    
    def _get_player_id(self, player_name):
        """Trouve l'ID - NE CRASH JAMAIS"""
        try:
            all_players = players.get_players()
            player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
            return player[0]['id'] if player else None
        except:
            return None
    
    def _get_base_game_logs(self, player_name, season):
        """Game logs - NE CRASH JAMAIS"""
        try:
            player_id = self._get_player_id(player_name)
            
            if player_id is None:
                return None
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            time.sleep(0.6)
            
            df = gamelog.get_data_frames()[0]
            
            # Colonnes essentielles
            required_cols = [
                'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 
                'PTS', 'AST', 'REB', 'FGM', 'FGA'
            ]
            
            # Vérifie que les colonnes existent
            available_cols = [col for col in required_cols if col in df.columns]
            
            if len(available_cols) < 5:
                return None
            
            df = df[available_cols].copy()
            
            # Parse date
            try:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            except:
                df['GAME_DATE'] = pd.to_datetime('2024-01-01')
            
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            # Minutes
            if 'MIN' in df.columns:
                df['MIN'] = df['MIN'].fillna('0:00').apply(self._parse_minutes)
            else:
                df['MIN'] = 30.0
            
            # Home/Away
            if 'MATCHUP' in df.columns:
                df['is_home'] = df['MATCHUP'].str.contains('vs.', na=False).astype(int)
                df['opponent'] = df['MATCHUP'].apply(self._extract_opponent)
            else:
                df['is_home'] = 1
                df['opponent'] = 'UNK'
            
            # Win/Loss
            if 'WL' in df.columns:
                df['won'] = (df['WL'] == 'W').astype(int)
            else:
                df['won'] = 0
            
            return df
            
        except Exception as e:
            print(f"❌ Game logs error: {e}")
            return None
    
    def _parse_minutes(self, min_str):
        """Parse minutes - NE CRASH JAMAIS"""
        try:
            if pd.isna(min_str) or min_str == '0:00':
                return 0.0
            parts = str(min_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        except:
            return 30.0
    
    def _extract_opponent(self, matchup):
        """Extract opponent - NE CRASH JAMAIS"""
        try:
            if pd.isna(matchup):
                return 'UNK'
            if 'vs.' in matchup:
                return matchup.split('vs.')[-1].strip()
            elif '@' in matchup:
                return matchup.split('@')[-1].strip()
            return 'UNK'
        except:
            return 'UNK'
    
    def _add_simple_features(self, df):
        """Ajoute features - NE CRASH JAMAIS"""
        try:
            # MOYENNES RÉCENTES
            for window in [3, 5, 10]:
                for stat in ['PTS', 'AST', 'REB', 'MIN']:
                    if stat in df.columns:
                        try:
                            df[f'avg_{stat.lower()}_last_{window}'] = df[stat].shift(1).rolling(window).mean().fillna(df[stat].mean())
                        except:
                            df[f'avg_{stat.lower()}_last_{window}'] = df[stat].mean() if stat in df.columns else 20.0
            
            # TENDANCES
            try:
                df['trend_pts_5'] = df['PTS'].shift(1).rolling(5).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
                ).fillna(0)
            except:
                df['trend_pts_5'] = 0
            
            # STABILITÉ
            try:
                df['std_pts_10'] = df['PTS'].shift(1).rolling(10).std().fillna(5.0)
            except:
                df['std_pts_10'] = 5.0
            
            # PERFORMANCE
            try:
                df['fg_pct'] = (df['FGM'] / df['FGA'] * 100).fillna(45.0)
            except:
                df['fg_pct'] = 45.0
            
            try:
                df['pts_per_min'] = (df['PTS'] / df['MIN']).fillna(0.7)
            except:
                df['pts_per_min'] = 0.7
            
            # FORME RÉCENTE
            try:
                df['recent_wins_5'] = df['won'].shift(1).rolling(5).sum().fillna(2.5)
                df['recent_win_pct'] = df['recent_wins_5'] / 5
            except:
                df['recent_wins_5'] = 2.5
                df['recent_win_pct'] = 0.5
            
            # REST DAYS
            try:
                df['rest_days'] = df['GAME_DATE'].diff().dt.days.abs().fillna(2.0).clip(0, 7)
                df['back_to_back'] = (df['rest_days'] == 1).astype(int)
            except:
                df['rest_days'] = 2.0
                df['back_to_back'] = 0
            
            # HOME/AWAY SPLIT
            try:
                home_avg = df[df['is_home'] == 1]['PTS'].mean()
                away_avg = df[df['is_home'] == 0]['PTS'].mean()
                df['home_avg_pts'] = home_avg if not pd.isna(home_avg) else df['PTS'].mean()
                df['away_avg_pts'] = away_avg if not pd.isna(away_avg) else df['PTS'].mean()
            except:
                df['home_avg_pts'] = df['PTS'].mean() if 'PTS' in df.columns else 20.0
                df['away_avg_pts'] = df['PTS'].mean() if 'PTS' in df.columns else 20.0
            
            # VS OPPONENT
            try:
                df['vs_opp_avg_pts'] = df.groupby('opponent')['PTS'].transform(
                    lambda x: x.shift(1).expanding().mean()
                ).fillna(df['PTS'].mean())
            except:
                df['vs_opp_avg_pts'] = df['PTS'].mean() if 'PTS' in df.columns else 20.0
            
            # GAME NUMBER
            try:
                df['games_played'] = range(len(df), 0, -1)
                df['season_fatigue'] = df['games_played'] / 82
            except:
                df['games_played'] = 30
                df['season_fatigue'] = 0.5
            
            # CONSISTENCY
            try:
                df['pts_consistency'] = 1 / (1 + df['std_pts_10'])
            except:
                df['pts_consistency'] = 0.2
            
            # INTERACTIONS
            try:
                df['min_x_fg_pct'] = df['MIN'] * df['fg_pct'] / 100
            except:
                df['min_x_fg_pct'] = 15.0
            
            try:
                df['rest_x_recent_win'] = df['rest_days'] * df['recent_win_pct']
            except:
                df['rest_x_recent_win'] = 1.0
            
            # POLYNOMIAL
            try:
                df['avg_pts_last_5_squared'] = df['avg_pts_last_5'] ** 2
            except:
                df['avg_pts_last_5_squared'] = 400.0
            
            # FILL ALL NAs
            df = df.fillna(0)
            
            return df
        
        except Exception as e:
            print(f"❌ Features error: {e}")
            return df
    
    def prepare_features_for_prediction(self, player_name, opponent='', is_home=True, current_features=None):
        """Prépare features - NE CRASH JAMAIS"""
        try:
            if current_features is not None and isinstance(current_features, pd.DataFrame):
                return current_features
            
            df = self.get_complete_player_data(player_name)
            
            if df is None or len(df) == 0:
                return None
            
            features = df.iloc[0:1].copy()
            features = features.select_dtypes(include=[np.number])
            features = features.drop(columns=['PTS', 'AST', 'REB'], errors='ignore')
            
            return features
        
        except Exception as e:
            print(f"❌ Prepare features error: {e}")
            return None


if __name__ == '__main__':
    collector = AdvancedDataCollector()
    df = collector.get_complete_player_data('LeBron James')
    
    if df is not None:
        print(f"\n✅ SUCCESS: {len(df)} games, {len(df.columns)} features")
    else:
        print("\n❌ FAILED")