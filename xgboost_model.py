#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NBA Model - VERSION AMÉLIORÉE
Avec: outlier removal, coefficient variation, skip haute variance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')


class XGBoostNBAModel:
    """Modèle XGBoost pour prédire stats NBA avec gestion outliers"""
    
    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.model = None
        self.scaler = StandardScaler()
        self.training_stats = {}
        
        # Hyperparamètres anti-overfitting renforcés
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 3,              # Réduit de 4 à 3
            'learning_rate': 0.05,       # Réduit de 0.1 à 0.05
            'n_estimators': 100,         # Réduit de 150 à 100
            'min_child_weight': 5,       # Augmenté de 3 à 5
            'subsample': 0.7,            # Réduit de 0.8 à 0.7
            'colsample_bytree': 0.7,     # Réduit de 0.8 à 0.7
            'reg_alpha': 2.0,            # Augmenté de 1.0 à 2.0
            'reg_lambda': 2.0,           # Augmenté de 1.0 à 2.0
            'random_state': 42
        }
    
    def remove_outliers(self, df, column, method='iqr', threshold=2.5):
        """
        Enlève les données aberrantes
        
        Methods:
        - 'iqr': Interquartile Range (conservateur)
        - 'zscore': Z-score (plus agressif)
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (df[column] >= lower) & (df[column] <= upper)
        
        elif method == 'zscore':
            z_scores = np.abs(scipy_stats.zscore(df[column]))
            mask = z_scores < threshold
        
        removed = len(df) - mask.sum()
        print(f"   🗑️  Outliers removed: {removed} ({removed/len(df)*100:.1f}%)")
        
        return df[mask].copy()
    
    def calculate_predictability_score(self, df, stat_col):
        """
        Calcule un score de prévisibilité (0-100)
        Basé sur: CV, stabilité, tendance
        """
        mean = df[stat_col].mean()
        std = df[stat_col].std()
        
        # Coefficient de variation (plus bas = plus stable)
        cv = (std / mean) * 100 if mean > 0 else 100
        
        # Stabilité (variance des différences)
        diffs = df[stat_col].diff().dropna()
        stability = diffs.std()
        
        # Tendance (est-ce qu'il y a un pattern?)
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[stat_col].values
        lr = LinearRegression().fit(X, y)
        trend_r2 = lr.score(X, y)
        
        # Score composite (0-100)
        cv_score = max(0, 100 - cv * 2)  # Pénalise CV élevé
        stability_score = max(0, 100 - stability * 10)
        trend_score = trend_r2 * 100
        
        total_score = (cv_score * 0.5 + stability_score * 0.3 + trend_score * 0.2)
        
        return {
            'score': round(total_score, 1),
            'cv': round(cv, 1),
            'stability': round(stability, 2),
            'trend_r2': round(trend_r2, 3),
            'is_predictable': total_score >= 40  # Seuil minimum
        }
    
    def train(self, player, season='2024-25', save_model=True):
        """Entraîne le modèle avec nettoyage outliers"""
        
        print(f"\n{'='*70}")
        print(f"🎯 TRAINING: {player} - {self.stat_type.upper()}")
        print(f"{'='*70}")
        
        # Import data collector
        from advanced_data_collector import AdvancedDataCollector
        collector = AdvancedDataCollector()
        
        # Récupère données
        df = collector.get_complete_player_data(player)
        if df is None or len(df) < 15:
            return {
                'status': 'ERROR',
                'message': f'Insufficient data: {len(df) if df is not None else 0} games'
            }
        
        stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}[self.stat_type]
        
        print(f"📊 Total games: {len(df)}")
        
        # ✅ 1. ENLÈVE LES OUTLIERS
        print(f"\n🧹 Removing outliers...")
        df_clean = self.remove_outliers(df, stat_col, method='iqr')
        print(f"   ✅ Clean games: {len(df_clean)}")
        
        if len(df_clean) < 15:
            return {
                'status': 'ERROR',
                'message': 'Too many outliers - insufficient clean data'
            }
        
        # ✅ 2. VÉRIFIE LA PRÉVISIBILITÉ
        pred_score = self.calculate_predictability_score(df_clean, stat_col)
        print(f"\n📈 Predictability Analysis:")
        print(f"   Score: {pred_score['score']}/100")
        print(f"   CV: {pred_score['cv']}%")
        print(f"   Stability: {pred_score['stability']}")
        print(f"   Trend R²: {pred_score['trend_r2']}")
        
        # ✅ 3. SKIP SI TROP INSTABLE
        if not pred_score['is_predictable']:
            print(f"\n⚠️  SKIPPED: Player too unpredictable (score < 40)")
            return {
                'status': 'SKIPPED',
                'message': f'Player too unpredictable for {self.stat_type}',
                'predictability_score': pred_score['score'],
                'cv': pred_score['cv']
            }
        
        # ✅ 4. FEATURES AVEC VARIANCE
        features_df = collector.prepare_features_for_prediction(player, '', True)
        
        # Ajoute coefficient de variation comme feature
        features_df['cv'] = pred_score['cv']
        features_df['stability'] = pred_score['stability']
        
        # Merge avec target
        df_clean = df_clean.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        df_clean['target'] = df_clean[stat_col]
        
        # Aligne features et target
        min_len = min(len(features_df), len(df_clean))
        features_df = features_df.iloc[:min_len]
        df_clean = df_clean.iloc[:min_len]
        
        X = features_df
        y = df_clean['target'].values
        
        print(f"\n🔢 Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        
        # ✅ 5. TRAIN/TEST SPLIT (30% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=False
        )
        
        print(f"\n📊 Split:")
        print(f"   Train: {len(X_train)} games")
        print(f"   Test: {len(X_test)} games")
        
        # ✅ 6. SCALE
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # ✅ 7. TRAIN avec early stopping
        self.model = xgb.XGBRegressor(**self.params)
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        # ✅ 8. METRICS
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        # Train metrics
        y_train_pred = self.model.predict(X_train_scaled)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        # Test metrics (VRAI performance!)
        y_test_pred = self.model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=min(3, len(X_train) // 10),
            scoring='r2'
        )
        
        overfitting = train_r2 - test_r2
        
        print(f"\n📊 RESULTS:")
        print(f"   Train R²: {train_r2:.3f}")
        print(f"   Test R²:  {test_r2:.3f} ⭐")
        print(f"   Test RMSE: {test_rmse:.2f}")
        print(f"   Overfitting: {overfitting:.3f}")
        print(f"   CV R² mean: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # ✅ 9. REJET SI TEST R² < 0.15
        if test_r2 < 0.15:
            print(f"\n❌ REJECTED: Test R² too low ({test_r2:.3f} < 0.15)")
            return {
                'status': 'REJECTED',
                'message': f'Model performs too poorly on test set',
                'test_r2': round(test_r2, 3),
                'predictability_score': pred_score['score']
            }
        
        # Store stats
        self.training_stats = {
            'player': player,
            'stat_type': self.stat_type,
            'train_metrics': {
                'r2': train_r2,
                'rmse': train_rmse
            },
            'test_metrics': {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            },
            'cv_results': {
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std()
            },
            'predictability': pred_score,
            'data': {
                'total_games': len(df),
                'clean_games': len(df_clean),
                'outliers_removed': len(df) - len(df_clean)
            },
            'stability': {
                'overfitting': overfitting,
                'cv_percent': cv_scores.std() / cv_scores.mean() * 100 if cv_scores.mean() > 0 else 100
            }
        }
        
        print(f"\n✅ Model trained successfully!")
        print(f"{'='*70}\n")
        
        return {
            'status': 'SUCCESS',
            'train_metrics': self.training_stats['train_metrics'],
            'test_metrics': self.training_stats['test_metrics'],
            'cv_results': self.training_stats['cv_results'],
            'predictability': pred_score,
            'stability': self.training_stats['stability']
        }
    
    def predict(self, features):
        """Fait une prédiction"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        
        # Confidence interval basé sur RMSE test
        test_rmse = self.training_stats['test_metrics']['rmse']
        std = test_rmse
        
        ci_lower = prediction - 1.96 * std
        ci_upper = prediction + 1.96 * std
        
        return {
            'prediction': float(prediction),
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            }
        }


class ModelManager:
    """Gère les modèles pour tous les joueurs"""
    
    def __init__(self):
        self.models = {}
    
    def predict(self, player, stat_type, opponent, is_home):
        """Prédiction avec gestion auto des modèles"""
        
        model_key = f"{player}_{stat_type}"
        
        # Entraîne si nécessaire
        if model_key not in self.models:
            print(f"🔄 Training new model: {model_key}")
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player, save_model=False)
            
            if result['status'] != 'SUCCESS':
                raise ValueError(f"Model training failed: {result.get('message', 'Unknown error')}")
            
            self.models[model_key] = model
        
        # Features pour prédiction
        from advanced_data_collector import AdvancedDataCollector
        collector = AdvancedDataCollector()
        features = collector.prepare_features_for_prediction(player, opponent, is_home)
        
        if features is None:
            raise ValueError("Unable to prepare features")
        
        # Prédiction
        return self.models[model_key].predict(features)


if __name__ == '__main__':
    # Test rapide
    model = XGBoostNBAModel(stat_type='points')
    result = model.train('LeBron James', '2024-25', save_model=False)
    
    print("\n" + "="*70)
    print("TEST RESULTS:")
    print("="*70)
    print(f"Status: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Test R²: {result['test_metrics']['r2']:.3f}")
        print(f"Predictability: {result['predictability']['score']}/100")
    print("="*70)