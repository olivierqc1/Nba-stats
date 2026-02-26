#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED DATA COLLECTOR - VERSION 10 VARIABLES + 3PT
Support complet: PTS, AST, REB, FG3M (3-pointers made)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

class AdvancedDataCollector:
    """Collector avec 10 variables + support 3PT"""
    
    def __init__(self):
        self.cache = {}
        
    def get_complete_player_data(self, player_name, season='2024-25'):
        """
        Récupère données player avec 10 VARIABLES + 3PT
        
        Stats collectées: PTS, AST, REB, FG3M (3-pointers made)
        """
        try:
            print(f"\n📥 Collecting {player_name}...")
            
            player_id = self._get_player_id(player_name)
            if player_id is None:
                print(f"❌ Player not found: {player_name}")
                return None
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            time.sleep(0.6)
            
            df = gamelog.get_data_frames()[0]
            
            if df is None or len(df) == 0:
                print(f"❌ No games found for {player_name}")
                return None
            
            print(f"   📊 Raw games: {len(df)}")
            
            # Colonnes essentielles + 3PT
            cols_needed = ['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'AST', 'REB', 'FG3M']
            
            available = [c for c in cols_needed if c in df.columns]
            if len(available) < 5:
                print(f"❌ Missing essential columns")
                return None
            
            df = df[available].copy()
            
            # Parse date
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            df = df.dropna(subset=['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            # Parse minutes
            if 'MIN' in df.columns:
                df['MIN'] = df['MIN'].fillna('0:00')
                df['MIN'] = df['MIN'].apply(self._parse_minutes)
            else:
                df['MIN'] = 30.0
            
            # Fill NaN pour stats (y compris 3PT)
            for col in ['PTS', 'AST', 'REB', 'FG3M']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)
                else:
                    df[col] = 0
            
            # Home/Away
            if 'MATCHUP' in df.columns:
                df['home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
            else:
                df['home'] = 0
            
            # Rest days
            df['rest_days'] = df['GAME_DATE'].diff(-1).dt.days.fillna(2)
            df['rest_days'] = df['rest_days'].clip(0, 7)
            
            # ============================================================
            # FEATURES OPTIMALES (10 VARIABLES) + 3PT
            # ============================================================
            
            # Moyennes 5 derniers matchs
            df['avg_pts_last_5'] = df['PTS'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_ast_last_5'] = df['AST'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_reb_last_5'] = df['REB'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_fg3m_last_5'] = df['FG3M'].shift(1).rolling(5, min_periods=1).mean()  # ← NOUVEAU
            
            # Moyennes 10 derniers matchs
            df['avg_pts_last_10'] = df['PTS'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_ast_last_10'] = df['AST'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_reb_last_10'] = df['REB'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_fg3m_last_10'] = df['FG3M'].shift(1).rolling(10, min_periods=1).mean()  # ← NOUVEAU
            
            # Minutes moyenne
            df['minutes_avg'] = df['MIN'].shift(1).rolling(10, min_periods=1).mean()
            
            # Opponent defensive rating (baseline)
            df['opponent_def_rating'] = 110.0
            
            # Pace (rythme de jeu)
            df['pace'] = 100.0
            
            # Usage rate
            total_production = df['PTS'] + df['AST'] + df['REB']
            df['usage_rate'] = (total_production * df['MIN']) / total_production.mean()
            df['usage_rate'] = df['usage_rate'].fillna(df['usage_rate'].mean())
            df['usage_rate'] = df['usage_rate'].clip(0, 200)
            
            # Back-to-back
            df['back_to_back'] = (df['rest_days'] == 0).astype(int)
            
            # Recent trend (pente des 5 derniers)
            df['recent_trend_pts'] = self._calculate_trend(df['PTS'], window=5)
            df['recent_trend_ast'] = self._calculate_trend(df['AST'], window=5)
            df['recent_trend_reb'] = self._calculate_trend(df['REB'], window=5)
            df['recent_trend_fg3m'] = self._calculate_trend(df['FG3M'], window=5)  # ← NOUVEAU
            
            # Fill NaN final
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)
            
            # Drop premiers matchs instables
            if len(df) > 15:
                df = df[10:].reset_index(drop=True)
                print(f"   🔧 Removed first 10 games (unstable features)")
            
            print(f"   ✅ Final: {len(df)} games, {len(df.columns)} features")
            print(f"   📊 Stats: PTS, AST, REB, 3PT (FG3M)")
            
            if len(df) < 15:
                print(f"   ⚠️  Only {len(df)} games - might be insufficient")
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR collecting {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_trend(self, series, window=5):
        """Calcule la pente (trend) des N derniers matchs"""
        def linear_slope(x):
            if len(x) < 2:
                return 0
            indices = np.arange(len(x))
            slope = np.polyfit(indices, x, 1)[0]
            return slope
        
        return series.shift(1).rolling(window, min_periods=2).apply(linear_slope, raw=False)
    
    def _get_player_id(self, player_name):
        """Trouve l'ID du joueur"""
        try:
            all_players = players.get_players()
            
            # Exact match
            player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
            if player:
                return player[0]['id']
            
            # Partial match
            player = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
            if player:
                return player[0]['id']
            
            return None
            
        except Exception as e:
            print(f"Error finding player ID: {e}")
            return None
    
    def _parse_minutes(self, min_str):
        """Parse minutes: "32:45" → 32.75"""
        try:
            if isinstance(min_str, (int, float)):
                return float(min_str)
            
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            
            return float(min_str)
        except:
            return 30.0
    
    def prepare_features_for_prediction(self, player_name, opponent='', is_home=True, current_features=None):
        """Prépare features pour prédiction"""
        try:
            if current_features is not None and isinstance(current_features, pd.DataFrame):
                return current_features
            
            df = self.get_complete_player_data(player_name)
            
            if df is None or len(df) == 0:
                return None
            
            features = df.iloc[0:1].copy()
            features = features.select_dtypes(include=[np.number])
            
            # Drop les stats cibles
            features = features.drop(columns=['PTS', 'AST', 'REB', 'FG3M'], errors='ignore')
            
            return features
        
        except Exception as e:
            print(f"❌ Prepare features error: {e}")
            return None


if __name__ == "__main__":
    collector = AdvancedDataCollector()
    
    print("\n" + "="*70)
    print("TEST: 10 VARIABLES + 3PT SUPPORT")
    print("="*70)
    
    df = collector.get_complete_player_data("Stephen Curry", "2024-25")
    
    if df is not None:
        print("\n✅ SUCCÈS!")
        print(f"\nMatchs: {len(df)}")
        print(f"\nFeatures disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nPremières lignes (aperçu stats 3PT):")
        print(df[['PTS', 'FG3M', 'avg_fg3m_last_5', 'avg_fg3m_last_10']].head())
    else:
        print("\n❌ ÉCHEC")
