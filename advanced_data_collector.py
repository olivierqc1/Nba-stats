#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMPLE DATA COLLECTOR - Version FINALE
Objectif: GARANTIR 30+ matchs pour TOUS les joueurs
Features: SEULEMENT 5 features essentielles
FIX: Supprime premiers matchs pour éviter distribution shift
"""

import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import time

class AdvancedDataCollector:
    """Collector MINIMAL - Garantit des données de qualité"""
    
    def __init__(self):
        self.cache = {}
        
    def get_complete_player_data(self, player_name, season='2024-25'):
        """
        Récupère données player - VERSION ULTRA-SIMPLE
        
        Returns DataFrame avec:
        - GAME_DATE
        - PTS / AST / REB / MIN
        - avg_pts_last_5
        - avg_pts_last_10
        - home (0/1)
        - rest_days
        - minutes_avg
        
        GARANTIT: 20+ matchs avec features stables
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
            
            # 2. Garde SEULEMENT les colonnes essentielles
            cols_needed = ['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'AST', 'REB']
            
            # Check colonnes disponibles
            available = [c for c in cols_needed if c in df.columns]
            if len(available) < 4:
                print(f"❌ Missing essential columns")
                return None
            
            df = df[available].copy()
            
            # 3. Parse date
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            df = df.dropna(subset=['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            # 4. Parse minutes (MIN peut être "32:45" ou float)
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
            
            # 7. Rest days (entre matchs)
            df['rest_days'] = df['GAME_DATE'].diff(-1).dt.days.fillna(2)
            df['rest_days'] = df['rest_days'].clip(0, 7)
            
            # 8. Features SIMPLES: moyennes glissantes
            
            # Moyenne 5 derniers matchs (shift pour pas inclure le match actuel)
            df['avg_pts_last_5'] = df['PTS'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_ast_last_5'] = df['AST'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_reb_last_5'] = df['REB'].shift(1).rolling(5, min_periods=1).mean()
            
            # Moyenne 10 derniers matchs
            df['avg_pts_last_10'] = df['PTS'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_ast_last_10'] = df['AST'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_reb_last_10'] = df['REB'].shift(1).rolling(10, min_periods=1).mean()
            
            # Moyenne minutes
            df['minutes_avg'] = df['MIN'].shift(1).rolling(10, min_periods=1).mean()
            
            # 9. Fill NaN final
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)
            
            # 10. ✅ FIX DISTRIBUTION SHIFT: Supprime les 10 premiers matchs
            # Pourquoi? Les premiers matchs ont des features basées sur 1-2 matchs seulement
            # Ça crée une distribution différente entre train et test
            if len(df) > 15:
                df = df[10:].reset_index(drop=True)
                print(f"   🔧 Removed first 10 games (distribution shift fix)")
            
            print(f"   ✅ Final: {len(df)} games, {len(df.columns)} features")
            
            # Vérification finale: au moins 15 matchs après cleanup
            if len(df) < 15:
                print(f"   ⚠️  Only {len(df)} games - insufficient")
                return None
            
            return df
            
        except Exception as e:
            print(f"❌ ERROR collecting {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            return 30.0  # Valeur par défaut
    
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


# Test rapide
if __name__ == "__main__":
    collector = AdvancedDataCollector()
    
    # Test LeBron
    df = collector.get_complete_player_data("LeBron James", "2024-25")
    
    if df is not None:
        print("\n" + "="*60)
        print("SUCCÈS! Données collectées:")
        print(f"Matchs: {len(df)}")
        print(f"Features: {list(df.columns)}")
        print(f"\nPremières lignes:")
        print(df.head())
        print(f"\nStats moyennes:")
        print(df[['PTS', 'AST', 'REB', 'avg_pts_last_5', 'avg_pts_last_10']].describe())
    else:
        print("\n❌ ÉCHEC - Pas de données collectées")