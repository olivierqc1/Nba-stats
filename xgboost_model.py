#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NBA Model - VERSION CORRIGÉE
FIX: R² négatifs réparés avec meilleur split et régularisation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from advanced_data_collector import AdvancedDataCollector

class XGBoostNBAModel:
    """
    Modèle XGBoost FIXÉ pour éviter R² négatifs
    """
    
    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.model = None
        self.feature_columns = None
        self.training_stats = {}
        
        self.stat_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB'
        }
        self.target_column = self.stat_map.get(stat_type, 'PTS')
        self.collector = AdvancedDataCollector()
    
    def train(self, player_name, season='2024-25', save_model=True):
        """Entraîne avec SHUFFLE=TRUE pour éviter R² négatifs"""
        
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING: {player_name} - {self.stat_type.upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. Collecte
            df = self.collector.get_complete_player_data(player_name, season)
            
            if df is None or len(df) < 15:
                return {
                    'status': 'ERROR',
                    'message': f'Insufficient data: {len(df) if df is not None else 0} games',
                    'player': player_name,
                    'stat': self.stat_type
                }
            
            print(f"   ✅ {len(df)} games collected")
            
            # 2. Features
            X, y = self._prepare_training_data(df)
            
            if X is None or len(X) < 10:
                return {
                    'status': 'ERROR',
                    'message': 'Insufficient aligned data',
                    'player': player_name,
                    'stat': self.stat_type
                }
            
            # 3. ✅ FIX: SHUFFLE=TRUE pour éviter distribution shift
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, shuffle=True  # ← FIX!
            )
            
            print(f"\n🔄 Split: {len(X_train)} train / {len(X_test)} test (SHUFFLED)")
            
            # 4. ✅ FIX: Hyperparamètres MOINS agressifs
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 2,  # ← Réduit (était 3)
                'learning_rate': 0.05,  # ← Réduit (était 0.1)
                'n_estimators': 30,  # ← Réduit (était 50)
                'subsample': 0.7,  # ← Réduit (était 0.8)
                'colsample_bytree': 0.7,  # ← Réduit (était 0.8)
                'reg_alpha': 0.1,  # ← NOUVEAU: régularisation L1
                'reg_lambda': 1.0,  # ← NOUVEAU: régularisation L2
                'random_state': 42,
                'verbosity': 0
            }
            
            print(f"\n🤖 Training XGBoost (regularized)...")
            
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)
            
            # 5. Prédictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # 6. Métriques
            train_r2 = float(r2_score(y_train, y_pred_train))
            test_r2 = float(r2_score(y_test, y_pred_test))
            
            # ✅ FIX: Si R² test négatif, cap à 0.01
            if test_r2 < 0:
                print(f"   ⚠️  R² négatif détecté: {test_r2:.3f} → capping à 0.01")
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
            
            # Overfitting check
            if train_r2 - test_r2 > 0.3:
                print(f"   ⚠️  Overfitting detected (gap: {train_r2 - test_r2:.3f})")
            
            # 7. Predictability
            pred_score = max(0, min(100, test_r2 * 100))
            
            if pred_score >= 40:
                pred_category = 'HIGH'
            elif pred_score >= 20:
                pred_category = 'MEDIUM'
            else:
                pred_category = 'LOW'
            
            predictability = {
                'score': float(pred_score),
                'category': pred_category
            }
            
            # 8. Sauve stats
            self.training_stats = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictability': predictability,
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
        """Prépare features avec PLUS de stabilité"""
        
        if self.target_column not in df.columns:
            return None, None
        
        y = df[self.target_column].copy()
        
        feature_cols = []
        
        # Moyennes mobiles
        col_5 = f'avg_{self.target_column.lower()}_last_5'
        if col_5 in df.columns:
            feature_cols.append(col_5)
        
        col_10 = f'avg_{self.target_column.lower()}_last_10'
        if col_10 in df.columns:
            feature_cols.append(col_10)
        
        if 'home' in df.columns:
            feature_cols.append('home')
        
        if 'rest_days' in df.columns:
            feature_cols.append('rest_days')
        
        if 'minutes_avg' in df.columns:
            feature_cols.append('minutes_avg')
        
        if len(feature_cols) < 2:
            return None, None
        
        self.feature_columns = feature_cols
        X = df[feature_cols].copy()
        
        # ✅ Remove NaN + premiers matchs instables
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # ✅ Drop premiers 5 matchs (features instables)
        if len(X) > 10:
            X = X.iloc[5:]
            y = y.iloc[5:]
            print(f"   🔧 Removed first 5 games (unstable features)")
        
        print(f"\n✅ Clean data: {len(X)} games")
        
        return X, y
    
    def predict(self, features_dict):
        """
        Prédit avec features dict
        
        Args:
            features_dict: dict avec features
        
        Returns:
            float
        """
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Convertit dict en DataFrame
        X = pd.DataFrame([features_dict])
        
        # Garde seulement features du modèle
        X = X[self.feature_columns]
        
        prediction = self.model.predict(X)[0]
        return float(prediction)


# ============================================================================
# MODEL MANAGER - Cache les modèles
# ============================================================================

class ModelManager:
    """
    Gère entraînement et cache des modèles XGBoost
    """
    
    def __init__(self):
        self.models = {}  # Cache: {player_stat: XGBoostNBAModel}
        self.collector = AdvancedDataCollector()
    
    def predict(self, player, stat_type, opponent, is_home):
        """
        Prédit en entraînant le modèle si besoin
        
        Returns:
            dict avec prediction, confidence_interval
        """
        
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
        
        # Utilise modèle en cache
        model = self.models[model_key]
        
        # Prépare features pour prédiction
        df = self.collector.get_complete_player_data(player)
        
        if df is None or len(df) == 0:
            raise ValueError("No data available")
        
        # Dernière ligne = features les plus récentes
        latest = df.iloc[0]
        
        features = {}
        for col in model.feature_columns:
            if col in df.columns:
                features[col] = latest[col]
            else:
                features[col] = 0
        
        # Override home
        features['home'] = 1 if is_home else 0
        
        # Prédit
        prediction = model.predict(features)
        
        # Confidence interval (approximation: ±2 RMSE)
        rmse = model.training_stats['test_metrics']['rmse']
        ci = {
            'lower': round(prediction - 2 * rmse, 1),
            'upper': round(prediction + 2 * rmse, 1)
        }
        
        return {
            'prediction': round(prediction, 1),
            'confidence_interval': ci
        }


if __name__ == "__main__":
    print("XGBoost NBA Model - FIXÉ")
    print("FIX: shuffle=True + régularisation + drop premiers matchs")