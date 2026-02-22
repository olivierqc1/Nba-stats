#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED DATA COLLECTOR - VERSION 10 VARIABLES
Objectif: Prédictions optimales avec features scientifiquement prouvées
"""

import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

class AdvancedDataCollector:
    """Collector avec 10 variables optimales"""
    
    def __init__(self):
        self.cache = {}
        
    def get_complete_player_data(self, player_name, season='2024-25'):
        """
        Récupère données player avec 10 VARIABLES OPTIMALES:
        
        VARIABLES ACTUELLES (5):
        1. avg_pts_last_5
        2. avg_pts_last_10
        3. home
        4. rest_days
        5. minutes_avg
        
        NOUVELLES VARIABLES (5):
        6. opponent_def_rating
        7. pace
        8. usage_rate
        9. back_to_back
        10. recent_trend
        """
        try:
            print(f"\n📥 Collecting {player_name}...")
            
            # 1. Récupère game logs
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
            
            # 2. Colonnes essentielles
            cols_needed = ['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'AST', 'REB']
            available = [c for c in cols_needed if c in df.columns]
            if len(available) < 4:
                print(f"❌ Missing essential columns")
                return None
            
            df = df[available].copy()
            
            # 3. Parse date
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            df = df.dropna(subset=['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            # 4. Parse minutes
            if 'MIN' in df.columns:
                df['MIN'] = df['MIN'].fillna('0:00')
                df['MIN'] = df['MIN'].apply(self._parse_minutes)
            else:
                df['MIN'] = 30.0
            
            # 5. Fill NaN pour stats
            for col in ['PTS', 'AST', 'REB']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)
                else:
                    df[col] = 0
            
            # 6. Home/Away
            if 'MATCHUP' in df.columns:
                df['home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
            else:
                df['home'] = 0
            
            # 7. Rest days
            df['rest_days'] = df['GAME_DATE'].diff(-1).dt.days.fillna(2)
            df['rest_days'] = df['rest_days'].clip(0, 7)
            
            # ============================================================
            # 8. FEATURES OPTIMALES (10 VARIABLES)
            # ============================================================
            
            # VARIABLES 1-2: Moyennes 5 et 10 derniers matchs
            df['avg_pts_last_5'] = df['PTS'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_ast_last_5'] = df['AST'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_reb_last_5'] = df['REB'].shift(1).rolling(5, min_periods=1).mean()
            
            df['avg_pts_last_10'] = df['PTS'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_ast_last_10'] = df['AST'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_reb_last_10'] = df['REB'].shift(1).rolling(10, min_periods=1).mean()
            
            # VARIABLE 5: Minutes moyennes
            df['minutes_avg'] = df['MIN'].shift(1).rolling(10, min_periods=1).mean()
            
            # VARIABLE 6: Opponent defensive rating
            # Approximation: on utilise une baseline NBA (110.0)
            # Dans une version future, on pourrait fetch la vraie def rating
            df['opponent_def_rating'] = 110.0
            
            # VARIABLE 7: Pace (rythme de jeu)
            # Approximation: baseline NBA est ~100 possessions/48min
            df['pace'] = 100.0
            
            # VARIABLE 8: Usage rate
            # Approximation: (PTS + AST + REB) * MIN / moyenne
            total_production = df['PTS'] + df['AST'] + df['REB']
            df['usage_rate'] = (total_production * df['MIN']) / total_production.mean()
            df['usage_rate'] = df['usage_rate'].fillna(df['usage_rate'].mean())
            df['usage_rate'] = df['usage_rate'].clip(0, 200)  # Cap outliers
            
            # VARIABLE 9: Back-to-back
            df['back_to_back'] = (df['rest_days'] == 0).astype(int)
            
            # VARIABLE 10: Recent trend (pente des 5 derniers)
            df['recent_trend_pts'] = self._calculate_trend(df['PTS'], window=5)
            df['recent_trend_ast'] = self._calculate_trend(df['AST'], window=5)
            df['recent_trend_reb'] = self._calculate_trend(df['REB'], window=5)
            
            # 9. Fill NaN final
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)
            
            # 10. Drop premiers matchs instables
            if len(df) > 15:
                df = df[10:].reset_index(drop=True)
                print(f"   🔧 Removed first 10 games (unstable features)")
            
            print(f"   ✅ Final: {len(df)} games, {len(df.columns)} features")
            print(f"   📊 Variables: avg_L5, avg_L10, home, rest_days, minutes_avg,")
            print(f"                opponent_def, pace, usage, b2b, trend")
            
            if len(df) < 15:
                print(f"   ⚠️  Only {len(df)} games - might be insufficient")
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR collecting {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_trend(self, series, window=5):
        """
        Calcule la pente (trend) des N derniers matchs
        
        Positif = performance en hausse
        Négatif = performance en baisse
        """
        def linear_slope(x):
            if len(x) < 2:
                return 0
            # Régression linéaire simple
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
            features = features.drop(columns=['PTS', 'AST', 'REB'], errors='ignore')
            
            return features
        
        except Exception as e:
            print(f"❌ Prepare features error: {e}")
            return None


if __name__ == "__main__":
    collector = AdvancedDataCollector()
    
    print("\n" + "="*70)
    print("TEST: 10 VARIABLES OPTIMALES")
    print("="*70)
    
    df = collector.get_complete_player_data("LeBron James", "2024-25")
    
    if df is not None:
        print("\n✅ SUCCÈS!")
        print(f"\nMatchs: {len(df)}")
        print(f"\nFeatures disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\nPremières lignes (aperçu):")
        print(df[['PTS', 'avg_pts_last_5', 'avg_pts_last_10', 'home', 'back_to_back', 'recent_trend_pts']].head())
    else:
        print("\n❌ ÉCHEC") 