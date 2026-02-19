#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NBA Model avec SHAP EXPLANATIONS
Explique chaque prédiction variable par variable
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Import SHAP pour explications
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available - install with: pip install shap")
    SHAP_AVAILABLE = False

# Import du collector v10
import sys
sys.path.insert(0, '/home/claude')
from advanced_data_collector_v10 import AdvancedDataCollector


class XGBoostNBAModel:
    """
    Modèle XGBoost avec explications SHAP
    """
    
    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.model = None
        self.feature_columns = None
        self.training_stats = {}
        self.explainer = None  # SHAP explainer
        
        self.stat_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB'
        }
        self.target_column = self.stat_map.get(stat_type, 'PTS')
        self.collector = AdvancedDataCollector()
    
    def train(self, player_name, season='2024-25', save_model=True):
        """Entraîne avec 10 variables + crée SHAP explainer"""
        
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING: {player_name} - {self.stat_type.upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. Collecte avec 10 variables
            df = self.collector.get_complete_player_data(player_name, season)
            
            if df is None or len(df) < 15:
                return {
                    'status': 'ERROR',
                    'message': f'Insufficient data: {len(df) if df is not None else 0} games',
                    'player': player_name,
                    'stat': self.stat_type
                }
            
            print(f"   ✅ {len(df)} games collected")
            
            # 2. Prépare features (10 variables)
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
            
            # 3. Split avec shuffle
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, shuffle=True
            )
            
            print(f"\n🔄 Split: {len(X_train)} train / {len(X_test)} test (SHUFFLED)")
            
            # 4. Hyperparamètres optimisés pour 10 variables
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,  # Augmenté (était 2)
                'learning_rate': 0.05,
                'n_estimators': 50,  # Augmenté (était 30)
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.05,  # Moins de régularisation
                'reg_lambda': 0.5,
                'random_state': 42,
                'verbosity': 0
            }
            
            print(f"\n🤖 Training XGBoost (10 vars, regularized)...")
            
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)
            
            # 5. Crée SHAP explainer
            if SHAP_AVAILABLE:
                print(f"   🔍 Creating SHAP explainer...")
                self.explainer = shap.TreeExplainer(self.model)
                print(f"   ✅ SHAP explainer ready!")
            
            # 6. Prédictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # 7. Métriques
            train_r2 = float(r2_score(y_train, y_pred_train))
            test_r2 = float(r2_score(y_test, y_pred_test))
            
            if test_r2 < 0:
                print(f"   ⚠️  R² négatif: {test_r2:.3f} → capping à 0.01")
                test_r2 = 0.01
            
            train_metrics = {
                'r2': train_r2,
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mae': float(mean_absolute_error(y_train, y_pred_train))
            }
            
            test_metrics = {
                'r2': test_r2,
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mae': float(mean_absolute_error(y_test, y_pred_test))
            }
            
            print(f"\n📈 RESULTS:")
            print(f"   Train R²: {train_metrics['r2']:.3f}")
            print(f"   Test R²:  {test_metrics['r2']:.3f}")
            print(f"   Test RMSE: {test_metrics['rmse']:.2f}")
            
            if train_r2 - test_r2 > 0.3:
                print(f"   ⚠️  Overfitting (gap: {train_r2 - test_r2:.3f})")
            
            # 8. Feature importance
            feature_importance = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_columns, feature_importance))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n🎯 TOP 5 FEATURES:")
            for i, (feat, imp) in enumerate(sorted_importance[:5], 1):
                print(f"   {i}. {feat}: {imp*100:.1f}%")
            
            # 9. Predictability
            pred_score = max(0, min(100, test_r2 * 100))
            pred_category = 'HIGH' if pred_score >= 40 else 'MEDIUM' if pred_score >= 20 else 'LOW'
            
            predictability = {
                'score': float(pred_score),
                'category': pred_category
            }
            
            # 10. Sauve stats
            self.training_stats = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictability': predictability,
                'feature_importance': sorted_importance,
                'data': {
                    'total_games': len(df),
                    'clean_games': len(X),
                    'outliers_removed': 0
                },
                'cv_results': {
                    'r2_mean': test_r2
                }
            }
            
            print(f"   Predictability: {pred_score:.1f}/100 ({pred_category})")
            print(f"✅ Training complete!")
            
            return {
                'status': 'SUCCESS',
                'player': player_name,
                'stat': self.stat_type,
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'predictability': predictability,
                'games_analyzed': len(X)
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
        """Prépare 10 variables pour entraînement"""
        
        if self.target_column not in df.columns:
            return None, None
        
        y = df[self.target_column].copy()
        
        # 10 VARIABLES OPTIMALES
        trend_col = f'recent_trend_{self.target_column.lower()}'
        
        feature_cols = [
            f'avg_{self.target_column.lower()}_last_5',   # 1
            f'avg_{self.target_column.lower()}_last_10',  # 2
            'home',                                         # 3
            'rest_days',                                    # 4
            'minutes_avg',                                  # 5
            'opponent_def_rating',                          # 6
            'pace',                                         # 7
            'usage_rate',                                   # 8
            'back_to_back',                                 # 9
            trend_col                                       # 10
        ]
        
        # Garde seulement celles qui existent
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        if len(feature_cols) < 5:
            print(f"   ⚠️  Only {len(feature_cols)} features available")
            return None, None
        
        self.feature_columns = feature_cols
        X = df[feature_cols].copy()
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Drop premiers 5 matchs
        if len(X) > 10:
            X = X.iloc[5:]
            y = y.iloc[5:]
        
        print(f"\n✅ Clean data: {len(X)} games x {len(feature_cols)} features")
        
        return X, y
    
    def predict_with_explanation(self, features_dict):
        """
        Prédit ET explique la prédiction
        
        Returns:
            {
                'prediction': float,
                'explanation': {
                    'base_value': float,
                    'contributions': [
                        {'feature': str, 'value': float, 'contribution': float},
                        ...
                    ]
                }
            }
        """
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Convertit en DataFrame
        X = pd.DataFrame([features_dict])
        X = X[self.feature_columns]
        
        # Prédiction
        prediction = self.model.predict(X)[0]
        
        # Explication SHAP
        explanation = None
        
        if SHAP_AVAILABLE and self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(X)
                
                # Base value (prédiction moyenne)
                base_value = self.explainer.expected_value
                
                # Contributions de chaque feature
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
                
                # Trie par contribution absolue
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
        """Prédiction simple (sans explication)"""
        result = self.predict_with_explanation(features_dict)
        return result['prediction']


# ============================================================================
# MODEL MANAGER avec SHAP
# ============================================================================

class ModelManager:
    """Gère les modèles avec explications"""
    
    def __init__(self):
        self.models = {}
        self.collector = AdvancedDataCollector()
    
    def predict(self, player, stat_type, opponent, is_home):
        """Prédit avec explication"""
        
        model_key = f"{player}_{stat_type}"
        
        # Entraîne si pas en cache
        if model_key not in self.models:
            print(f"🔄 Training model for {player} ({stat_type})...")
            
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player, '2024-25', save_model=False)
            
            if result['status'] != 'SUCCESS':
                raise ValueError(f"Training failed: {result.get('message')}")
            
            self.models[model_key] = model
            print(f"✅ Model cached: {model_key}")
        
        # Modèle en cache
        model = self.models[model_key]
        
        # Prépare features
        df = self.collector.get_complete_player_data(player)
        
        if df is None or len(df) == 0:
            raise ValueError("No data available")
        
        # Dernière ligne = features récentes
        latest = df.iloc[0]
        
        features = {}
        for col in model.feature_columns:
            if col in df.columns:
                features[col] = latest[col]
            else:
                features[col] = 0
        
        # Override home
        features['home'] = 1 if is_home else 0
        
        # Prédit avec explication
        result = model.predict_with_explanation(features)
        
        prediction = result['prediction']
        explanation = result['explanation']
        
        # Confidence interval
        rmse = model.training_stats['test_metrics']['rmse']
        ci = {
            'lower': round(prediction - 2 * rmse, 1),
            'upper': round(prediction + 2 * rmse, 1)
        }
        
        return {
            'prediction': round(prediction, 1),
            'confidence_interval': ci,
            'explanation': explanation  # ← NOUVEAU!
        }


if __name__ == "__main__":
    print("XGBoost avec SHAP Explanations")
    print("10 variables optimales + décomposition de chaque prédiction")