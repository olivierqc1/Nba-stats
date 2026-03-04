#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBOOST MODEL - VERSION AMELIOREE
Changements cles:
- Utilise les nouvelles features (over_rate, momentum, exp_avg, line_value)
- Filtre les features a faible variance
- Cross-validation temporelle (TimeSeriesSplit)
- Anti-overfitting renforce (max_depth=3, reg plus fort)
- Score de confiance base sur la consistance du joueur
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


FEATURE_SETS = {
    'points': [
        'avg_pts_last_5', 'avg_pts_last_10', 'exp_avg_pts',
        'recent_over_rate_pts', 'pts_momentum', 'pts_per_min',
        'recent_trend_pts', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'minutes_consistency', 'usage_rate'
    ],
    'assists': [
        'avg_ast_last_5', 'avg_ast_last_10', 'exp_avg_ast',
        'recent_over_rate_ast', 'ast_momentum', 'ast_per_min',
        'recent_trend_ast', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'minutes_consistency', 'usage_rate'
    ],
    'rebounds': [
        'avg_reb_last_5', 'avg_reb_last_10', 'exp_avg_reb',
        'recent_over_rate_reb', 'reb_momentum', 'reb_per_min',
        'recent_trend_reb', 'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'minutes_consistency', 'usage_rate'
    ],
    '3pt': [
        'avg_fg3m_last_5', 'avg_fg3m_last_10',
        'recent_trend_fg3m',
        'opponent_def_rating', 'pace',
        'home', 'rest_days', 'back_to_back',
        'minutes_avg', 'usage_rate'
    ]
}

XGBOOST_PARAMS = {
    'max_depth':        3,
    'learning_rate':    0.05,
    'n_estimators':     80,
    'subsample':        0.75,
    'colsample_bytree': 0.75,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'min_child_weight': 3,
    'random_state':     42,
    'verbosity':        0
}


class XGBoostNBAModel:

    def __init__(self, player_name, stat_type):
        self.player_name   = player_name
        self.stat_type     = stat_type
        self.model         = None
        self.feature_cols  = []
        self.training_stats = {}
        self.trained       = False

    def _select_available_features(self, df):
        wanted = FEATURE_SETS.get(self.stat_type, FEATURE_SETS['points'])
        available = [f for f in wanted if f in df.columns]
        # Enlever features a variance quasi-nulle
        if len(df) > 5:
            variances = df[available].var()
            available = [f for f in available if variances.get(f, 0) > 0.01]
        return available

    def train(self, df):
        TARGET_MAP = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB', '3pt': 'FG3M'}
        target_col = TARGET_MAP.get(self.stat_type, 'PTS')

        if target_col not in df.columns:
            return False
        if len(df) < 15:
            return False

        # Ordre chronologique pour validation temporelle
        df_sorted = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

        self.feature_cols = self._select_available_features(df_sorted)
        if len(self.feature_cols) < 4:
            print(f"Not enough features ({len(self.feature_cols)}) for {self.player_name}")
            return False

        X = df_sorted[self.feature_cols].fillna(0)
        y = df_sorted[target_col]

        # Récupère les sample_weights si disponibles (calculés par advanced_data_collector)
        weights = df_sorted['sample_weight'].values if 'sample_weight' in df_sorted.columns else None

        # TimeSeriesSplit — pas de data leakage
        n_splits = min(3, len(df_sorted) // 10)
        if n_splits < 2:
            split_idx = int(len(df_sorted) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            w_train = weights[:split_idx] if weights is not None else None
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(tscv.split(X))
            train_idx, test_idx = splits[-1]
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]
            w_train = weights[train_idx] if weights is not None else None

        self.model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        # sample_weight: matchs extremes pesent moins fort, matchs normaux = 1.0
        self.model.fit(X_train, y_train, sample_weight=w_train)

        y_pred_test  = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)

        test_r2   = float(r2_score(y_test, y_pred_test))
        train_r2  = float(r2_score(y_train, y_pred_train))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        # Alerte overfitting
        overfit_gap = train_r2 - test_r2
        if overfit_gap > 0.3:
            print(f"   WARNING overfitting gap={overfit_gap:.2f} for {self.player_name} {self.stat_type}")

        self.training_stats = {
            'test_metrics':  {'r2': round(test_r2, 3),  'rmse': round(test_rmse, 2)},
            'train_metrics': {'r2': round(train_r2, 3)},
            'overfit_gap':   round(overfit_gap, 3),
            'n_features':    len(self.feature_cols),
            'n_games':       len(df_sorted),
            'feature_importance': self._get_feature_importance()
        }

        self.trained = True
        print(f"   {self.player_name} {self.stat_type}: R²={test_r2:.3f} RMSE={test_rmse:.2f} overfit={overfit_gap:.2f}")
        return True

    def _get_feature_importance(self):
        if self.model is None:
            return {}
        scores = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_cols, [round(float(s), 4) for s in scores]),
            key=lambda x: x[1], reverse=True
        ))

    def predict(self, features_df):
        if self.model is None or not self.trained:
            return None
        try:
            available = [f for f in self.feature_cols if f in features_df.columns]
            X = features_df[available].fillna(0)
            for f in self.feature_cols:
                if f not in X.columns:
                    X[f] = 0.0
            X = X[self.feature_cols]
            prediction = float(self.model.predict(X)[0])
            return round(prediction, 1)
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


class ModelManager:

    def __init__(self):
        self.models     = {}
        self.collectors = {}

    def _get_collector(self):
        from advanced_data_collector import AdvancedDataCollector
        if 'default' not in self.collectors:
            self.collectors['default'] = AdvancedDataCollector()
        return self.collectors['default']

    def get_or_train(self, player_name, stat_type, df=None):
        key = f"{player_name}_{stat_type}"
        if key in self.models and self.models[key].trained:
            return self.models[key]

        if df is None:
            collector = self._get_collector()
            df = collector.get_complete_player_data(player_name)

        if df is None or len(df) < 15:
            return None

        model = XGBoostNBAModel(player_name, stat_type)
        if model.train(df):
            self.models[key] = model
            return model
        return None

    def predict(self, player_name, stat_type, opponent='', is_home=True):
        """Retourne prediction + intervalle de confiance."""
        collector = self._get_collector()
        df = collector.get_complete_player_data(player_name)

        if df is None or len(df) < 15:
            return {'prediction': None, 'error': 'Not enough data'}

        model = self.get_or_train(player_name, stat_type, df)
        if model is None:
            return {'prediction': None, 'error': 'Model training failed'}

        # Features du match a predire (ligne la plus recente = dernier match connu)
        features = df.iloc[0:1].copy()
        features = features.select_dtypes(include=[np.number])
        features = features.drop(columns=['PTS', 'AST', 'REB', 'FG3M'], errors='ignore')

        prediction = model.predict(features)
        if prediction is None:
            return {'prediction': None, 'error': 'Prediction failed'}

        rmse = model.training_stats.get('test_metrics', {}).get('rmse', 4.0)
        ci_z = 1.96
        return {
            'prediction': prediction,
            'confidence_interval': {
                'lower': round(prediction - ci_z * rmse, 1),
                'upper': round(prediction + ci_z * rmse, 1)
            },
            'model_r2':   model.training_stats.get('test_metrics', {}).get('r2', 0),
            'model_rmse': rmse,
            'overfit_gap': model.training_stats.get('overfit_gap', 0),
            'explanation': model.training_stats.get('feature_importance', {})
        }
