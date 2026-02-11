#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NBA Model - VERSION COMPATIBLE BACKEND
Interface correcte: train(player_name, season, save_model)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from advanced_data_collector import AdvancedDataCollector

class XGBoostNBAModel:
    """
    Modèle XGBoost avec interface compatible backend
    
    Interface attendue:
    - __init__(stat_type)
    - train(player_name, season, save_model) → returns dict
    """
    
    def __init__(self, stat_type='points'):
        """
        Args:
            stat_type: 'points', 'assists', ou 'rebounds'
        """
        self.stat_type = stat_type
        self.model = None
        self.feature_columns = None
        self.training_stats = {}
        
        # Map stat_type to column name
        self.stat_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB'
        }
        self.target_column = self.stat_map.get(stat_type, 'PTS')
        
        # Collector
        self.collector = AdvancedDataCollector()
    
    def train(self, player_name, season='2025-26', save_model=True):
        """
        Entraîne le modèle sur un joueur
        
        Args:
            player_name: Nom du joueur
            season: Saison (ex: '2024-25')
            save_model: Sauve le modèle (ignoré pour l'instant)
        
        Returns:
            dict avec:
            {
                'status': 'SUCCESS' | 'ERROR',
                'test_metrics': {'r2': ..., 'rmse': ..., 'mae': ...},
                'train_metrics': {'r2': ..., 'rmse': ..., 'mae': ...},
                'predictability': {'score': ..., 'category': ...},
                'message': ...
            }
        """
        
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING: {player_name} - {self.stat_type.upper()}")
        print(f"{'='*60}")
        
        try:
            # 1. Collecte données
            print(f"\n📥 Collecting data for {player_name}...")
            df = self.collector.get_complete_player_data(player_name, season)
            
            if df is None or len(df) == 0:
                return {
                    'status': 'ERROR',
                    'message': f'Insufficient aligned data',
                    'player': player_name,
                    'stat': self.stat_type
                }
            
            print(f"   ✅ {len(df)} games collected")
            
            # 2. Prépare features
            X, y = self._prepare_training_data(df)
            
            if X is None or len(X) < 10:
                return {
                    'status': 'ERROR',
                    'message': 'Insufficient aligned data',
                    'player': player_name,
                    'stat': self.stat_type
                }
            
            # 3. Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            print(f"\n🔄 Split: {len(X_train)} train / {len(X_test)} test")
            
            # 4. Entraîne XGBoost
            print(f"\n🤖 Training XGBoost...")
            
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 50,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 0
            }
            
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train, y_train)
            
            # 5. Prédictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # 6. Métriques
            train_metrics = {
                'r2': float(r2_score(y_train, y_pred_train)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                'mae': float(mean_absolute_error(y_train, y_pred_train))
            }
            
            test_metrics = {
                'r2': float(r2_score(y_test, y_pred_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mae': float(mean_absolute_error(y_test, y_pred_test))
            }
            
            print(f"\n📈 RESULTS:")
            print(f"   Train R²: {train_metrics['r2']:.3f}")
            print(f"   Test R²:  {test_metrics['r2']:.3f}")
            print(f"   Test RMSE: {test_metrics['rmse']:.2f}")
            
            # 7. Predictability score (basé sur R²)
            pred_score = max(0, min(100, test_metrics['r2'] * 100))
            
            if pred_score >= 50:
                pred_category = 'HIGH'
            elif pred_score >= 30:
                pred_category = 'MEDIUM'
            else:
                pred_category = 'LOW'
            
            predictability = {
                'score': float(pred_score),
                'category': pred_category
            }
            
            # 8. Sauve stats pour usage ultérieur
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
                    'r2_mean': test_metrics['r2']  # Approximation
                }
            }
            
            print(f"   Predictability: {pred_score:.1f}/100 ({pred_category})")
            print(f"\n✅ Training complete!")
            
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
        """
        Prépare X (features) et y (target) pour entraînement
        
        Features:
        - avg_X_last_5
        - avg_X_last_10
        - home
        - rest_days
        - minutes_avg
        """
        
        # Target column
        if self.target_column not in df.columns:
            print(f"❌ Target column '{self.target_column}' not found")
            return None, None
        
        y = df[self.target_column].copy()
        
        # Features
        feature_cols = []
        
        # Moyenne 5 matchs
        col_5 = f'avg_{self.target_column.lower()}_last_5'
        if col_5 in df.columns:
            feature_cols.append(col_5)
        
        # Moyenne 10 matchs
        col_10 = f'avg_{self.target_column.lower()}_last_10'
        if col_10 in df.columns:
            feature_cols.append(col_10)
        
        # Home/Away
        if 'home' in df.columns:
            feature_cols.append('home')
        
        # Rest days
        if 'rest_days' in df.columns:
            feature_cols.append('rest_days')
        
        # Minutes moyennes
        if 'minutes_avg' in df.columns:
            feature_cols.append('minutes_avg')
        
        if len(feature_cols) < 2:
            print(f"❌ Insufficient features: {len(feature_cols)}")
            return None, None
        
        self.feature_columns = feature_cols
        
        print(f"\n📊 Features ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"   {i}. {col}")
        
        X = df[feature_cols].copy()
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"\n✅ Clean data: {len(X)} games (removed {len(df) - len(X)} with NaN)")
        
        return X, y
    
    def predict(self, df_new):
        """
        Prédit sur nouvelles données
        
        Args:
            df_new: DataFrame avec features
        
        Returns:
            float: Prédiction
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if self.feature_columns is None:
            raise ValueError("Features not defined!")
        
        # Prepare features
        X_new = df_new[self.feature_columns].copy()
        
        # Fill NaN if any
        if X_new.isna().any().any():
            X_new = X_new.fillna(X_new.mean())
        
        # Predict
        prediction = self.model.predict(X_new)[0]
        
        return float(prediction)


# Alias pour compatibilité
ModelManager = XGBoostNBAModel


# Test
if __name__ == "__main__":
    print("XGBoost NBA Model - Backend Compatible")
    print("Interface: train(player_name, season, save_model)")
    print("Returns: dict with status, test_metrics, train_metrics, predictability")