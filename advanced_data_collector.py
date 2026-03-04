#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED DATA COLLECTOR v3
- Nouvelles features: over_rate, momentum, exp_avg, line_value
- Filtre matchs hors normes (< 20 min, outliers statistiques)
- Winsorisation des valeurs extremes
"""

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players, teams
import time


class AdvancedDataCollector:

    def __init__(self):
        self.cache             = {}
        self._team_stats_cache = {}

    # ------------------------------------------------------------------
    # TEAM DEFENSIVE STATS
    # ------------------------------------------------------------------

    def get_team_defensive_stats(self, season='2024-25'):
        key = f'team_stats_{season}'
        if key in self._team_stats_cache:
            return self._team_stats_cache[key]
        try:
            print("   Fetching team defensive stats...")
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_simple_defense='Advanced',
                per_mode_simple='PerGame'
            )
            time.sleep(0.6)
            df     = stats.get_data_frames()[0]
            result = {}
            all_t  = teams.get_teams()
            for _, row in df.iterrows():
                match = [t for t in all_t if t['full_name'] == row['TEAM_NAME']]
                abbr  = match[0]['abbreviation'] if match else row['TEAM_NAME'][:3].upper()
                result[abbr] = {
                    'def_rating': float(row.get('DEF_RATING', 112.0)),
                    'pace':       float(row.get('PACE',       99.0))
                }
            self._team_stats_cache[key] = result
            return result
        except Exception as e:
            print(f"   Team stats fallback: {e}")
            return {t['abbreviation']: {'def_rating': 112.0, 'pace': 99.0}
                    for t in teams.get_teams()}

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _get_player_id(self, player_name):
        try:
            all_p      = players.get_players()
            name_clean = player_name.strip().lower()

            # 1. Correspondance exacte
            exact = [p for p in all_p if p['full_name'].lower() == name_clean]
            if exact:
                return exact[0]['id']

            # 2. Nom contenu dans le nom complet
            partial = [p for p in all_p if name_clean in p['full_name'].lower()]
            if partial:
                return partial[0]['id']

            # 3. Tous les mots du nom entré présents dans le nom complet
            #    ex: "keynote george" → tokens ["keynote","george"]
            #    → matche "Keyonte George" si suffisamment proche
            tokens = name_clean.split()
            token_match = [
                p for p in all_p
                if all(t in p['full_name'].lower() for t in tokens)
            ]
            if token_match:
                return token_match[0]['id']

            # 4. Matching flou sur le nom de famille uniquement
            #    Utile pour fautes de frappe légères (keynote → keyonte)
            if len(tokens) >= 2:
                last_name = tokens[-1]
                last_match = [
                    p for p in all_p
                    if last_name in p['last_name'].lower()
                ]
                if len(last_match) == 1:
                    return last_match[0]['id']
                # Si plusieurs résultats, essayer de matcher aussi le prénom approx.
                if len(last_match) > 1 and len(tokens[0]) >= 3:
                    first_prefix = tokens[0][:3]
                    refined = [
                        p for p in last_match
                        if p['first_name'].lower().startswith(first_prefix)
                    ]
                    if refined:
                        return refined[0]['id']

            print(f"Player not found: '{player_name}'")
            return None
        except Exception as e:
            print(f"Player ID error: {e}")
            return None

    def _parse_minutes(self, v):
        try:
            if isinstance(v, (int, float)):
                return float(v)
            if ':' in str(v):
                p = str(v).split(':')
                return float(p[0]) + float(p[1]) / 60
            return float(v)
        except Exception:
            return 30.0

    def _get_opp_abbr(self, matchup):
        try:
            return str(matchup).strip().split()[-1].upper()
        except Exception:
            return 'UNK'

    def _calculate_trend(self, series, window=5):
        def slope(x):
            if len(x) < 2:
                return 0.0
            return float(np.polyfit(np.arange(len(x)), x, 1)[0])
        return series.shift(1).rolling(window, min_periods=2).apply(slope, raw=False)

    def _exp_avg_rolling(self, series, span=5):
        result = []
        vals   = series.values
        for i in range(len(vals)):
            window = vals[max(0, i - span):i]
            if len(window) == 0:
                result.append(float(np.nanmean(vals)) if len(vals) > 0 else 0.0)
                continue
            w = np.exp(np.linspace(0, 1, len(window)))
            result.append(float(np.average(window, weights=w / w.sum())))
        return pd.Series(result, index=series.index)

    # ------------------------------------------------------------------
    # FILTRE MATCHS HORS NORMES
    # ------------------------------------------------------------------

    def _filter_outlier_games(self, df):
        """
        Au lieu de supprimer ou capper les matchs extremes:
        1. Filtre uniquement les matchs < 20 min (vraiment inutilisables)
        2. Calcule un sample_weight pour chaque match:
           - Match normal (dans ±2σ) → poids 1.0
           - Match hors norme (> 2σ) → poids réduit proportionnellement
           - Plus le match est extreme, moins il pèse (min 0.15)
        Le modèle XGBoost utilisera ces poids lors du fit().
        """
        initial_count = len(df)

        # 1. Filtre minutes uniquement (DNP, blowout < 20 min)
        df_clean = df[df['MIN'] >= 20].copy()

        excluded = initial_count - len(df_clean)
        if excluded > 0:
            print(f"   Outlier filter: {excluded} matchs exclus (< 20 min)")

        # 2. Calcul du sample_weight par match
        #    Basé sur l'écart aux stats principales (PTS, AST, REB)
        weights = np.ones(len(df_clean))

        for col in ['PTS', 'AST', 'REB']:
            if col not in df_clean.columns:
                continue
            mean = df_clean[col].mean()
            std  = df_clean[col].std()
            if std < 0.5:
                continue
            z_scores = ((df_clean[col] - mean) / std).abs().values
            # Au-delà de 2σ: poids = max(0.15, 1 - 0.25 * (z - 2))
            # Exemple: z=2 → poids 1.0 | z=3 → 0.75 | z=4 → 0.5 | z=5+ → 0.15
            col_weights = np.where(
                z_scores <= 2.0,
                1.0,
                np.maximum(0.15, 1.0 - 0.25 * (z_scores - 2.0))
            )
            weights = weights * col_weights  # combine les 3 dimensions

        # Normalise pour que la moyenne reste à 1.0
        weights = weights / weights.mean()
        df_clean['sample_weight'] = weights

        outlier_count = (weights < 0.8).sum()
        if outlier_count > 0:
            print(f"   Weighting: {outlier_count} matchs extremes pondérés à la baisse")

        return df_clean, excluded

    # ------------------------------------------------------------------
    # MAIN DATA COLLECTION
    # ------------------------------------------------------------------

    def get_complete_player_data(self, player_name, season='2024-25'):
        try:
            print(f"\nCollecting {player_name} [{season}]...")
            player_id = self._get_player_id(player_name)
            if player_id is None:
                print(f"Player not found: {player_name}")
                return None

            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            time.sleep(0.6)
            df = gamelog.get_data_frames()[0]

            if df is None or len(df) == 0:
                return None

            cols = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'AST', 'REB', 'FG3M']
            df   = df[[c for c in cols if c in df.columns]].copy()
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
            df = df.dropna(subset=['GAME_DATE'])

            # Plus recent en premier
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

            df['MIN'] = df['MIN'].fillna('0:00').apply(self._parse_minutes)

            for col in ['PTS', 'AST', 'REB', 'FG3M']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                else:
                    df[col] = 0.0

            df['home']         = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
            df['opponent_abbr'] = df['MATCHUP'].apply(self._get_opp_abbr)
            df['rest_days']    = df['GAME_DATE'].diff(-1).dt.days.fillna(2).clip(0, 7)
            df['back_to_back'] = (df['rest_days'] == 0).astype(int)

            # FILTRE HORS NORMES
            df, games_excluded = self._filter_outlier_games(df)
            df['games_excluded_count'] = games_excluded  # stocke pour l'API

            if len(df) < 10:
                print(f"   Pas assez de matchs apres filtre: {len(df)}")
                return None

            # Stats defensives
            team_stats = self.get_team_defensive_stats(season)
            df['opponent_def_rating'] = df['opponent_abbr'].apply(
                lambda a: team_stats.get(a, {}).get('def_rating', 112.0)
            )
            df['pace'] = df['opponent_abbr'].apply(
                lambda a: team_stats.get(a, {}).get('pace', 99.0)
            )

            # Moyennes glissantes standard
            for col, tag in [('PTS','pts'), ('AST','ast'), ('REB','reb'), ('FG3M','fg3m')]:
                df[f'avg_{tag}_last_5']  = df[col].shift(1).rolling(5,  min_periods=1).mean()
                df[f'avg_{tag}_last_10'] = df[col].shift(1).rolling(10, min_periods=1).mean()

            # Minutes
            df['minutes_avg']         = df['MIN'].shift(1).rolling(10, min_periods=1).mean()
            df['minutes_consistency'] = df['MIN'].shift(1).rolling(10, min_periods=2).std().fillna(5.0)

            # Usage rate
            total = df['PTS'] + df['AST'] + df['REB']
            df['usage_rate'] = ((total * df['MIN']) / total.mean().clip(1)).clip(0, 200).fillna(0)

            # Trends
            df['recent_trend_pts']  = self._calculate_trend(df['PTS'],  5)
            df['recent_trend_ast']  = self._calculate_trend(df['AST'],  5)
            df['recent_trend_reb']  = self._calculate_trend(df['REB'],  5)
            df['recent_trend_fg3m'] = self._calculate_trend(df['FG3M'], 5)

            # Per minute
            safe_min = df['MIN'].replace(0, np.nan)
            df['pts_per_min'] = (df['PTS'] / safe_min).fillna(0).clip(0, 3)
            df['ast_per_min'] = (df['AST'] / safe_min).fillna(0).clip(0, 2)
            df['reb_per_min'] = (df['REB'] / safe_min).fillna(0).clip(0, 2)

            # Moyennes exponentielles
            df['exp_avg_pts'] = self._exp_avg_rolling(df['PTS'].shift(1).fillna(df['PTS'].mean()))
            df['exp_avg_ast'] = self._exp_avg_rolling(df['AST'].shift(1).fillna(df['AST'].mean()))
            df['exp_avg_reb'] = self._exp_avg_rolling(df['REB'].shift(1).fillna(df['REB'].mean()))

            # Momentum
            df['pts_momentum'] = df['avg_pts_last_5'] - df['avg_pts_last_10']
            df['ast_momentum'] = df['avg_ast_last_5'] - df['avg_ast_last_10']
            df['reb_momentum'] = df['avg_reb_last_5'] - df['avg_reb_last_10']

            # Recent over rate
            for col, tag in [('PTS','pts'), ('AST','ast'), ('REB','reb')]:
                avg = df[col].mean()
                df[f'recent_over_rate_{tag}'] = (
                    df[col].shift(1)
                    .rolling(10, min_periods=3)
                    .apply(lambda x: (x > avg).mean(), raw=True)
                    .fillna(0.5)
                )

            # Fill NaN final
            num_cols = df.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                m = df[c].mean()
                df[c] = df[c].fillna(m if pd.notna(m) else 0.0)

            if len(df) > 10:
                df = df.iloc[5:].reset_index(drop=True)

            print(f"   {len(df)} matchs, {len(df.columns)} features, {games_excluded} exclus")
            return df

        except Exception as e:
            print(f"ERROR {player_name}: {e}")
            import traceback; traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # LINE VALUE SCORE
    # ------------------------------------------------------------------

    def get_line_value_score(self, player_name, stat_type, bookmaker_line, season='2024-25'):
        df = self.get_complete_player_data(player_name, season)
        if df is None or len(df) < 5:
            return 0.0
        col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}.get(stat_type, 'PTS')
        if col not in df.columns:
            return 0.0
        avg5  = float(df[col].head(5).mean())
        avg10 = float(df[col].head(10).mean())
        wtd   = avg5 * 0.7 + avg10 * 0.3
        std   = float(df[col].head(15).std())
        if std < 0.5:
            return 0.0
        return round((wtd - bookmaker_line) / std, 3)

    # ------------------------------------------------------------------
    # PREPARE FEATURES
    # ------------------------------------------------------------------

    def prepare_features_for_prediction(self, player_name, opponent='', is_home=True, current_features=None):
        try:
            if current_features is not None and isinstance(current_features, pd.DataFrame):
                return current_features
            df = self.get_complete_player_data(player_name)
            if df is None or len(df) == 0:
                return None
            features = df.iloc[0:1].copy().select_dtypes(include=[np.number])
            return features.drop(columns=['PTS', 'AST', 'REB', 'FG3M'], errors='ignore')
        except Exception as e:
            print(f"Prepare features error: {e}")
            return None
