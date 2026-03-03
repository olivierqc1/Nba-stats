#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NBA Model avec SHAP EXPLANATIONS
CORRECTIONS:
1. train_test_split avec shuffle=False → split TEMPOREL correct
   (les matchs récents sont dans le test set, pas mélangés avec le train)
2. Garde l'ordre chronologique pour évaluation réaliste
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available - install with: pip install shap")
    SHAP_AVAILABLE = False

import sys
sys.path.insert(0, '/home/claude')
from advanced_data_collector import AdvancedDataCollector


class XGBoostNBAModel:

    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.model = None
        self.feature_columns = None
        self.training_stats = {}
        self.explainer = None

        self.stat_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB',
            '3pt': 'FG3M'
        }
        self.target_column = self.stat_map.get(stat_type, 'PTS')
        self.collector = AdvancedDataCollector()

    def train(self, player_name, season='2024-25', save_model=True):
        """
        Entraîne avec split TEMPOREL correct.
        CORRECTION CLÉ: shuffle=False → les 75% premiers matchs (les plus
        anciens) forment le train set, les 25% plus récents le test set.
        C'est la seule façon d'évaluer honnêtement un modèle de prédiction.
        """

        print(f"\n{'='*60}")
        print(f"🎯 TRAINING: {player_name} - {self.stat_type.upper()}")
        print(f"{'='*60}")

        try:
            df = self.collector.get_complete_player_data(player_name, season)

            if df is None or len(df) < 15:
                return {
                    'status': 'ERROR',
                    'message': f'Insufficient data: {len(df) if df is not None else 0} games',
                    'player': player_name,
                    'stat': self.stat_type
                }

            print(f"   ✅ {len(df)} games collected")

            X, y = self._prepare_training_data(df)

            if X is None or len(X) < 10:
                return {
                    'status': 'ERROR',
                    'message': 'Insufficient aligned data',
                    'player': player_name,
                    'stat': self.stat_type
                }

            print(f"\n📊 Features ({len(self.feature_columns)}):")
            for i, col in enumerate(self.feature_columns, 1):
                print(f"   {i}. {col}")

            # ================================================================
            # CORRECTION PRINCIPALE: shuffle=False
            # Les données sont déjà triées chronologiquement (plus récent en
            # premier dans le DataFrame, mais _prepare_training_data les
            # remet en ordre croissant). Le test set = matchs les plus récents.
            # ================================================================
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.25,
                random_state=42,
                shuffle=False   # ← CORRECTION: était True
            )

            print(f"\n🔄 Split TEMPOREL: {len(X_train)} train (anciens) / {len(X_test)} test (récents)")
            print(f"   ⚠️  Note: R² test sera probablement plus bas qu'avant.")
            print(f"   C'est NORMAL et HONNÊTE — c'est la vraie performance du modèle.")

            params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,
                'learning_rate': 0.05,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.05,
                'reg_lambda': 0.5,
                'random_state': 42,
                'verbosity': 0
            }

            print(f"\n🤖 Training XGBoost...")

            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)

            if SHAP_AVAILABLE:
                print(f"   🔍 Creating SHAP explainer...")
                self.explainer = shap.TreeExplainer(self.model)
                print(f"   ✅ SHAP explainer ready!")

            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            train_r2 = float(r2_score(y_train, y_pred_train))
            test_r2 = float(r2_score(y_test, y_pred_test))

            # R² négatif = modèle pire que la moyenne — c'est une info utile,
            # on la garde visible plutôt que de la cacher
            if test_r2 < 0:
                print(f"   ⚠️  R² TEST NÉGATIF: {test_r2:.3f}")
                print(f"       → Le modèle est moins bon que prédire la moyenne.")
                print(f"       → NE PAS parier sur ce joueur/stat combo.")
                # On cap à 0 pour l'edge calculation mais on log l'avertissement
                test_r2_capped = 0.01
            else:
                test_r2_capped = test_r2

            train_metrics = {
                'r2': train_r2,
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mae': float(mean_absolute_error(y_train, y_pred_train))
            }

            test_metrics = {
                'r2': test_r2,           # Valeur réelle (peut être négative)
                'r2_display': test_r2_capped,  # Valeur affichée (min 0.01)
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mae': float(mean_absolute_error(y_test, y_pred_test))
            }

            print(f"\n📈 RESULTS (SPLIT TEMPOREL HONNÊTE):")
            print(f"   Train R²: {train_metrics['r2']:.3f}")
            print(f"   Test R²:  {test_metrics['r2']:.3f}  ← valeur réelle")
            print(f"   Test RMSE: {test_metrics['rmse']:.2f}")

            gap = train_r2 - max(test_r2, 0)
            if gap > 0.3:
                print(f"   ⚠️  Overfitting détecté (gap train-test: {gap:.3f})")
                print(f"       → Considère réduire max_depth ou augmenter reg_lambda")
            elif gap > 0.15:
                print(f"   ⚡ Léger overfitting (gap: {gap:.3f}) — acceptable")
            else:
                print(f"   ✅ Bon fit (gap: {gap:.3f})")

            feature_importance = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_columns, feature_importance))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            print(f"\n🎯 TOP 5 FEATURES:")
            for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
                print(f"   {i}. {feat}: {imp*100:.1f}%")

            # Si opponent_def_rating n'est pas dans le top 5, c'est suspect
            opp_rank = next((i+1 for i, (f, _) in enumerate(sorted_importance) 
                            if f == 'opponent_def_rating'), None)
            if opp_rank:
                print(f"   📍 opponent_def_rating rank: #{opp_rank}")

            pred_score = max(0, min(100, test_r2_capped * 100))
            pred_category = 'HIGH' if pred_score >= 40 else 'MEDIUM' if pred_score >= 20 else 'LOW'

            self.training_stats = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictability': {'score': float(pred_score), 'category': pred_category},
                'feature_importance': sorted_importance,
                'data': {
                    'total_games': len(df),
                    'clean_games': len(X),
                    'outliers_removed': 0
                },
                'cv_results': {
                    'r2_mean': test_r2_capped
                },
                'split_type': 'TEMPORAL'  # Pour tracer l'origine
            }

            return {
                'status': 'SUCCESS',
                'player': player_name,
                'stat': self.stat_type,
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'predictability': {'score': float(pred_score), 'category': pred_category},
                'games_analyzed': len(X),
                'split_type': 'TEMPORAL'
            }

        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'ERROR',
                'message': str(e),
                'player': player_name,
                'stat': self.stat_type
            }

    def _prepare_training_data(self, df):
        """
        Prépare 10 variables pour entraînement.
        IMPORTANT: remet les données en ordre CHRONOLOGIQUE (plus ancien en
        premier) pour que le split temporel soit correct.
        """

        if self.target_column not in df.columns:
            return None, None

        # ================================================================
        # CORRECTION: remet en ordre chronologique (ascendant)
        # Le DataFrame arrive trié descendant (plus récent en premier)
        # Pour le split temporel, on veut: anciens → train, récents → test
        # ================================================================
        df_sorted = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)

        y = df_sorted[self.target_column].copy()

        if self.target_column == 'PTS':
            trend_col = 'recent_trend_pts'
        elif self.target_column == 'AST':
            trend_col = 'recent_trend_ast'
        elif self.target_column == 'REB':
            trend_col = 'recent_trend_reb'
        elif self.target_column == 'FG3M':
            trend_col = 'recent_trend_fg3m'
        else:
            trend_col = f'recent_trend_{self.target_column.lower()}'

        feature_cols = [
            f'avg_{self.target_column.lower()}_last_5',
            f'avg_{self.target_column.lower()}_last_10',
            'home',
            'rest_days',
            'minutes_avg',
            'opponent_def_rating',   # ← Maintenant RÉEL
            'pace',                  # ← Maintenant RÉEL
            'usage_rate',
            'back_to_back',
            trend_col
        ]

        feature_cols = [c for c in feature_cols if c in df_sorted.columns]

        if len(feature_cols) < 5:
            print(f"   ⚠️  Only {len(feature_cols)} features available")
            return None, None

        self.feature_columns = feature_cols
        X = df_sorted[feature_cols].copy()

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        # Drop les 5 premiers matchs (rolling instable)
        if len(X) > 10:
            X = X.iloc[5:]
            y = y.iloc[5:]

        print(f"\n✅ Clean data: {len(X)} games x {len(feature_cols)} features")
        print(f"   Split: premiers {int(len(X)*0.75)} matchs → train, "
              f"derniers {len(X)-int(len(X)*0.75)} matchs → test")

        return X, y

    def predict_with_explanation(self, features_dict):
        if self.model is None:
            raise ValueError("Model not trained!")

        X = pd.DataFrame([features_dict])
        X = X[self.feature_columns]

        prediction = self.model.predict(X)[0]

        explanation = None
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X)
                base_value = self.explainer.expected_value

                contributions = []
                for i, (feat, val) in enumerate(features_dict.items()):
                    if feat in self.feature_columns:
                        idx = self.feature_columns.index(feat)
                        contrib = float(shap_values[0][idx])
                        contributions.append({
                            'feature': feat,
                            'value': float(val),
                            'contribution': contrib
                        })

                contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
                explanation = {
                    'base_value': float(base_value),
                    'contributions': contributions
                }

            except Exception as e:
                print(f"⚠️  SHAP explanation failed: {e}")

        return {
            'prediction': float(prediction),
            'explanation': explanation
        }

    def predict(self, features_dict):
        result = self.predict_with_explanation(features_dict)
        return result['prediction']


class ModelManager:

    def __init__(self):
        self.models = {}
        self.collector = AdvancedDataCollector()

    def predict(self, player, stat_type, opponent, is_home):

        model_key = f"{player}_{stat_type}"

        if model_key not in self.models:
            print(f"🔄 Training model for {player} ({stat_type})...")

            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player, '2024-25', save_model=False)

            if result['status'] != 'SUCCESS':
                raise ValueError(f"Training failed: {result.get('message')}")

            self.models[model_key] = model
            print(f"✅ Model cached: {model_key}")

        model = self.models[model_key]

        df = self.collector.get_complete_player_data(player)

        if df is None or len(df) == 0:
            raise ValueError("No data available")

        latest = df.iloc[0]

        features = {}
        for col in model.feature_columns:
            if col in df.columns:
                features[col] = latest[col]
            else:
                features[col] = 0

        features['home'] = 1 if is_home else 0

        # Override opponent_def_rating avec la vraie valeur de l'adversaire
        team_stats = self.collector.get_team_defensive_stats()
        opp_upper = opponent.upper()
        if opp_upper in team_stats:
            features['opponent_def_rating'] = team_stats[opp_upper]['def_rating']
            features['pace'] = team_stats[opp_upper]['pace']
            print(f"   ✅ Real stats for {opp_upper}: "
                  f"DEF={features['opponent_def_rating']:.1f}, "
                  f"PACE={features['pace']:.1f}")

        result = model.predict_with_explanation(features)

        prediction = result['prediction']
        explanation = result['explanation']

        rmse = model.training_stats['test_metrics']['rmse']
        ci = {
            'lower': round(prediction - 2 * rmse, 1),
            'upper': round(prediction + 2 * rmse, 1)
        }

        return {
            'prediction': round(prediction, 1),
            'confidence_interval': ci,
            'explanation': explanation
        }


if __name__ == "__main__":
    print("XGBoost — Split TEMPOREL + Vraies stats adversaires")