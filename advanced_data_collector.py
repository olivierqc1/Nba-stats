#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED DATA COLLECTOR - VERSION CORRIGÉE
Corrections:
1. opponent_def_rating RÉEL (via LeagueDashTeamStats)
2. pace RÉEL (via LeagueDashTeamStats)
3. FG3M support conservé
"""

import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, teamgamelog
from nba_api.stats.static import players, teams
import time

class AdvancedDataCollector:
    
    def __init__(self):
        self.cache = {}
        self._team_stats_cache = {}  # Cache pour éviter appels répétés

    # =========================================================================
    # NOUVEAU: Récupère les vraies stats défensives et pace de chaque équipe
    # =========================================================================

    def get_team_defensive_stats(self, season='2024-25'):
        """
        Récupère defensive rating et pace pour toutes les équipes NBA.
        Retourne un dict: { 'BOS': {'def_rating': 108.2, 'pace': 98.1}, ... }
        """
        cache_key = f'team_stats_{season}'
        if cache_key in self._team_stats_cache:
            return self._team_stats_cache[cache_key]

        try:
            print("   📡 Fetching real team defensive stats...")

            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_simple_defense='Advanced',
                per_mode_simple='PerGame'
            )
            time.sleep(0.6)

            df = stats.get_data_frames()[0]

            # Colonnes disponibles: TEAM_ID, TEAM_NAME, DEF_RATING, PACE, etc.
            result = {}
            for _, row in df.iterrows():
                team_name = row['TEAM_NAME']

                # Trouve l'abréviation
                all_teams = teams.get_teams()
                team_match = [t for t in all_teams if t['full_name'] == team_name]
                abbr = team_match[0]['abbreviation'] if team_match else team_name[:3].upper()

                result[abbr] = {
                    'def_rating': float(row.get('DEF_RATING', 112.0)),
                    'pace': float(row.get('PACE', 99.0))
                }

            print(f"   ✅ Team stats loaded for {len(result)} teams")
            self._team_stats_cache[cache_key] = result
            return result

        except Exception as e:
            print(f"   ⚠️  Could not fetch team stats: {e} — using league averages")
            # Fallback: moyennes de la ligue 2024-25 (approx.)
            fallback = {}
            all_teams = teams.get_teams()
            for t in all_teams:
                fallback[t['abbreviation']] = {
                    'def_rating': 112.0,
                    'pace': 99.0
                }
            return fallback

    def _get_opponent_abbr(self, matchup_str):
        """
        Extrait l'abréviation de l'adversaire depuis la chaîne MATCHUP.
        Ex: 'LAL vs. GSW' → 'GSW'
            'LAL @ BOS'   → 'BOS'
        """
        try:
            parts = matchup_str.strip().split()
            # Format: "TEAM vs. OPP" ou "TEAM @ OPP"
            return parts[-1].upper()
        except:
            return 'UNK'

    # =========================================================================
    # PRINCIPAL: collecte données joueur
    # =========================================================================

    def get_complete_player_data(self, player_name, season='2024-25'):
        """
        Récupère données player avec 10 VARIABLES + 3PT
        CORRECTION: opponent_def_rating et pace sont maintenant RÉELS
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

            # Fill NaN stats
            for col in ['PTS', 'AST', 'REB', 'FG3M']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)
                else:
                    df[col] = 0

            # Home/Away
            if 'MATCHUP' in df.columns:
                df['home'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
                df['opponent_abbr'] = df['MATCHUP'].apply(self._get_opponent_abbr)
            else:
                df['home'] = 0
                df['opponent_abbr'] = 'UNK'

            # Rest days
            df['rest_days'] = df['GAME_DATE'].diff(-1).dt.days.fillna(2)
            df['rest_days'] = df['rest_days'].clip(0, 7)

            # ================================================================
            # CORRECTION PRINCIPALE: opponent_def_rating et pace RÉELS
            # ================================================================
            team_stats = self.get_team_defensive_stats(season)

            df['opponent_def_rating'] = df['opponent_abbr'].apply(
                lambda abbr: team_stats.get(abbr, {}).get('def_rating', 112.0)
            )
            df['pace'] = df['opponent_abbr'].apply(
                lambda abbr: team_stats.get(abbr, {}).get('pace', 99.0)
            )

            print(f"   ✅ Real def ratings: min={df['opponent_def_rating'].min():.1f}, "
                  f"max={df['opponent_def_rating'].max():.1f}, "
                  f"std={df['opponent_def_rating'].std():.2f}")

            # Moyennes 5 derniers matchs
            df['avg_pts_last_5'] = df['PTS'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_ast_last_5'] = df['AST'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_reb_last_5'] = df['REB'].shift(1).rolling(5, min_periods=1).mean()
            df['avg_fg3m_last_5'] = df['FG3M'].shift(1).rolling(5, min_periods=1).mean()

            # Moyennes 10 derniers matchs
            df['avg_pts_last_10'] = df['PTS'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_ast_last_10'] = df['AST'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_reb_last_10'] = df['REB'].shift(1).rolling(10, min_periods=1).mean()
            df['avg_fg3m_last_10'] = df['FG3M'].shift(1).rolling(10, min_periods=1).mean()

            # Minutes moyenne
            df['minutes_avg'] = df['MIN'].shift(1).rolling(10, min_periods=1).mean()

            # Usage rate
            total_production = df['PTS'] + df['AST'] + df['REB']
            df['usage_rate'] = (total_production * df['MIN']) / total_production.mean()
            df['usage_rate'] = df['usage_rate'].fillna(df['usage_rate'].mean())
            df['usage_rate'] = df['usage_rate'].clip(0, 200)

            # Back-to-back
            df['back_to_back'] = (df['rest_days'] == 0).astype(int)

            # Recent trend
            df['recent_trend_pts'] = self._calculate_trend(df['PTS'], window=5)
            df['recent_trend_ast'] = self._calculate_trend(df['AST'], window=5)
            df['recent_trend_reb'] = self._calculate_trend(df['REB'], window=5)
            df['recent_trend_fg3m'] = self._calculate_trend(df['FG3M'], window=5)

            # Fill NaN final
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].mean() if len(df[col].dropna()) > 0 else 0)

            # Drop premiers matchs instables (pas assez d'historique pour les rolling)
            if len(df) > 15:
                df = df[10:].reset_index(drop=True)
                print(f"   🔧 Removed first 10 games (unstable features)")

            print(f"   ✅ Final: {len(df)} games, {len(df.columns)} features")
            print(f"   📊 Stats: PTS, AST, REB, 3PT | Def ratings: REAL")

            if len(df) < 15:
                print(f"   ⚠️  Only {len(df)} games - might be insufficient")

            return df

        except Exception as e:
            print(f"❌ ERROR collecting {player_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_trend(self, series, window=5):
        def linear_slope(x):
            if len(x) < 2:
                return 0
            indices = np.arange(len(x))
            slope = np.polyfit(indices, x, 1)[0]
            return slope
        return series.shift(1).rolling(window, min_periods=2).apply(linear_slope, raw=False)

    def _get_player_id(self, player_name):
        try:
            all_players = players.get_players()
            player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
            if player:
                return player[0]['id']
            player = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
            if player:
                return player[0]['id']
            return None
        except Exception as e:
            print(f"Error finding player ID: {e}")
            return None

    def _parse_minutes(self, min_str):
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
        try:
            if current_features is not None and isinstance(current_features, pd.DataFrame):
                return current_features
            df = self.get_complete_player_data(player_name)
            if df is None or len(df) == 0:
                return None
            features = df.iloc[0:1].copy()
            features = features.select_dtypes(include=[np.number])
            features = features.drop(columns=['PTS', 'AST', 'REB', 'FG3M'], errors='ignore')
            return features
        except Exception as e:
            print(f"❌ Prepare features error: {e}")
            return None


if __name__ == "__main__":
    collector = AdvancedDataCollector()

    print("\n" + "="*70)
    print("TEST: REAL OPPONENT DEF RATINGS + PACE")
    print("="*70)

    df = collector.get_complete_player_data("Stephen Curry", "2024-25")

    if df is not None:
        print("\n✅ SUCCÈS!")
        print(f"\nMatchs: {len(df)}")
        print(f"\nVariation opponent_def_rating (doit être > 0):")
        print(df['opponent_def_rating'].describe())
        print(f"\nVariation pace (doit être > 0):")
        print(df['pace'].describe())
    else:
        print("\n❌ ÉCHEC")