#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMPLE XGBOOST MODEL
Utilise SEULEMENT 5-8 features pour maximiser la stabilité
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

class ModelManager:
    """Modèle SIMPLE - Features minimales pour stabilité maximale"""
    
    def __init__(self, stat_type='points'):
        """
        Args:
            stat_type: 'points', 'assists', ou 'rebounds'
        """
        self.stat_type = stat_type
        self.model = None
        self.feature_columns = None
        self.target_column = None
        
        # Map stat_type to column name
        stat_map = {
            'points': 'PTS',
            'assists': 'AST',
            'rebounds': 'REB'
        }
        self.target_column = stat_map.get(stat_type, 'PTS')
        
    def prepare_features(self, df):
        """
        Sélectionne SEULEMENT les features essentielles
        
        Features utilisées (5-8 max):
        1. avg_X_last_5 (moyenne 5 derniers matchs)
        2. avg_X_last_10 (moyenne 10 derniers matchs)
        3. home (domicile=1, extérieur=0)
        4. rest_days (jours de repos)
        5. minutes_avg (minutes moyennes)
        
        Optionnel selon disponibilité:
        6. MIN (minutes ce match - si dispo en prédiction)
        """
        feature_cols = []
        
        # Feature 1: Moyenne 5 derniers matchs (ESSENTIELLE)
        col_5 = f'avg_{self.target_column.lower()}_last_5'
        if col_5 in df.columns:
            feature_cols.append(col_5)
        
        # Feature 2: Moyenne 10 derniers matchs (ESSENTIELLE)
        col_10 = f'avg_{self.target_column.lower()}_last_10'
        if col_10 in df.columns:
            feature_cols.append(col_10)
        
        # Feature 3: Home/Away
        if 'home' in df.columns:
            feature_cols.append('home')
        
        # Feature 4: Rest days
        if 'rest_days' in df.columns:
            feature_cols.append('rest_days')
        
        # Feature 5: Minutes moyennes
        if 'minutes_avg' in df.columns:
            feature_cols.append('minutes_avg')
        
        # Feature 6 (optionnel): Minutes ce match
        if 'MIN' in df.columns:
            feature_cols.append('MIN')
        
        self.feature_columns = feature_cols
        
        print(f"\n📊 Features utilisées ({len(feature_cols)}):")
        for i, col in enumerate(feature_cols, 1):
            print(f"   {i}. {col}")
        
        return df[feature_cols].copy()
    
    def train(self, df, test_size=0.2, random_state=42):
        """
        Entraîne le modèle XGBoost
        
        Args:
            df: DataFrame avec features + target
            test_size: Proportion du test set
            random_state: Seed pour reproductibilité
        
        Returns:
            dict avec métriques (R², MAE, RMSE)
        """
        
        # Vérification données minimales
        if df is None or len(df) < 10:
            print(f"❌ Insufficient data: {len(df) if df is not None else 0} games")
            return None
        
        # Target
        if self.target_column not in df.columns:
            print(f"❌ Target column '{self.target_column}' not found")
            return None
        
        y = df[self.target_column].copy()
        
        # Features
        X = self.prepare_features(df)
        
        if len(X.columns) < 2:
            print(f"❌ Insufficient features: {len(X.columns)}")
            return None
        
        # Remove NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"\n🎯 Training data: {len(X)} games")
        print(f"   Target: {self.target_column} (mean={y.mean():.1f}, std={y.std():.1f})")
        
        if len(X) < 10:
            print(f"❌ After cleanup: only {len(X)} games remaining")
            return None
        
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
        
        # XGBoost params - SIMPLES pour éviter overfitting
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,  # Shallow trees
            'learning_rate': 0.1,
            'n_estimators': 50,  # Peu d'arbres
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'verbosity': 0
        }
        
        # Train
        print(f"\n🤖 Training XGBoost (max_depth=3, n_est=50)...")
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n📈 RÉSULTATS:")
        print(f"   R² Train: {r2_train:.3f}")
        print(f"   R² Test:  {r2_test:.3f}")
        print(f"   MAE Test: {mae_test:.2f}")
        print(f"   RMSE Test: {rmse_test:.2f}")
        
        # Feature importance
        importance = self.model.feature_importances_
        print(f"\n🎯 Feature Importance:")
        for feat, imp in sorted(zip(self.feature_columns, importance), key=lambda x: x[1], reverse=True):
            print(f"   {feat}: {imp:.3f}")
        
        results = {
            'r2_train': r2_train,
            'r2_test': r2_test,
            'mae': mae_test,
            'rmse': rmse_test,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features_used': len(self.feature_columns),
            'feature_names': self.feature_columns,
            'feature_importance': dict(zip(self.feature_columns, importance))
        }
        
        return results
    
    def predict(self, df_new):
        """
        Prédit sur nouvelles données
        
        Args:
            df_new: DataFrame avec mêmes features que training
        
        Returns:
            float: Prédiction
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if self.feature_columns is None:
            raise ValueError("Features not defined!")
        
        # Prepare features
        X_new = df_new[self.feature_columns].copy()
        
        # Check NaN
        if X_new.isna().any().any():
            print(f"⚠️  NaN in features - filling with mean")
            X_new = X_new.fillna(X_new.mean())
        
        # Predict
        prediction = self.model.predict(X_new)[0]
        
        return prediction
    
    def analyze_opportunity(self, df, bookmaker_line):
        """
        Analyse si une opportunité existe
        
        Args:
            df: DataFrame avec historique du joueur
            bookmaker_line: Ligne du bookmaker (float)
        
        Returns:
            dict avec status, prediction, edge, etc.
        """
        
        # 1. Train model
        print(f"\n{'='*60}")
        print(f"🎲 ANALYSE: {self.stat_type.upper()}")
        print(f"   Bookmaker line: {bookmaker_line}")
        
        metrics = self.train(df)
        
        if metrics is None:
            return {
                'status': 'ERROR',
                'message': 'Insufficient aligned data',
                'player': 'Unknown',
                'stat': self.stat_type
            }
        
        # 2. Check R² test
        r2_test = metrics['r2_test']
        
        print(f"\n📊 Model Quality:")
        print(f"   R² Test: {r2_test:.3f}")
        
        # Seuil PERMISSIF: R² >= 0.20
        if r2_test < 0.20:
            print(f"   ❌ REJECTED: R² too low (< 0.20)")
            return {
                'status': 'REJECTED',
                'reason': 'Low R²',
                'r2_test': r2_test,
                'message': f'R² test = {r2_test:.3f} < 0.20'
            }
        
        # 3. Predict sur dernier match (row 0 = plus récent)
        last_game = df.iloc[[0]]
        prediction = self.predict(last_game)
        
        print(f"\n🎯 Prediction:")
        print(f"   Model: {prediction:.1f}")
        print(f"   Line: {bookmaker_line:.1f}")
        print(f"   Diff: {prediction - bookmaker_line:+.1f}")
        
        # 4. Calculate edge
        edge = abs(prediction - bookmaker_line) / bookmaker_line * 100
        
        # 5. Recommendation
        if abs(prediction - bookmaker_line) < 0.5:
            recommendation = 'SKIP'
            reason = 'Too close to line'
        elif prediction > bookmaker_line:
            recommendation = 'OVER'
            reason = f'+{prediction - bookmaker_line:.1f} edge'
        else:
            recommendation = 'UNDER'
            reason = f'{prediction - bookmaker_line:.1f} edge'
        
        print(f"\n💡 Recommendation: {recommendation}")
        print(f"   Reason: {reason}")
        print(f"   Edge: {edge:.1f}%")
        
        return {
            'status': 'SUCCESS',
            'prediction': float(prediction),
            'bookmaker_line': float(bookmaker_line),
            'edge': float(edge),
            'recommendation': recommendation,
            'reason': reason,
            'r2_test': float(r2_test),
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'games_analyzed': metrics['train_size'] + metrics['test_size'],
            'features_used': metrics['features_used']
        }


# Alias pour compatibilité avec différents imports du backend
# Le backend peut importer soit ModelManager soit XGBoostNBAModel
XGBoostNBAModel = ModelManager  # Alias principal


# Test
if __name__ == "__main__":
    print("Simple XGBoost Model - Ready")
    print("Features: avg_last_5, avg_last_10, home, rest_days, minutes_avg")
    print("Target R² test: >= 0.20")
    print("Max features: 5-8")