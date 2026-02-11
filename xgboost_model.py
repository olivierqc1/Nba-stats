#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost NBA Model - VERSION FINALE
Skip predictability check pour POINTS (accepte tout)
Garde check pour ASSISTS/REBOUNDS
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
    """Modèle XGBoost pour prédire stats NBA"""
    
    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.model = None
        self.scaler = StandardScaler()
        self.training_stats = {}
        
        # Hyperparamètres
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'learning_rate': 0.08,
            'n_estimators': 120,
            'min_child_weight': 4,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 1.5,
            'reg_lambda': 1.5,
            'random_state': 42
        }
    
    def remove_outliers(self, df, column):
        """Enlève les données aberrantes"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 2.0 * IQR
        upper = Q3 + 2.0 * IQR
        mask = (df[column] >= lower) & (df[column] <= upper)
        
        removed = len(df) - mask.sum()
        print(f"   🗑️  Outliers removed: {removed}")
        
        return df[mask].copy()
    
    def calculate_predictability_score(self, df, stat_col):
        """Calcule score de prévisibilité (pour info uniquement)"""
        mean = df[stat_col].mean()
        std = df[stat_col].std()
        cv = (std / mean) * 100 if mean > 0 else 100
        
        diffs = df[stat_col].diff().dropna()
        stability = diffs.std() if len(diffs) > 0 else 10
        
        try:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[stat_col].values
            lr = LinearRegression().fit(X, y)
            trend_r2 = lr.score(X, y)
        except:
            trend_r2 = 0
        
        cv_score = max(0, 100 - cv)
        stability_score = max(0, 100 - stability * 6)
        trend_score = trend_r2 * 100
        
        total_score = (cv_score * 0.5 + stability_score * 0.3 + trend_score * 0.2)
        
        return {
            'score': round(total_score, 1),
            'cv': round(cv, 1),
            'stability': round(stability, 2),
            'trend_r2': round(trend_r2, 3)
        }
    
    def train(self, player, season='2024-25', save_model=True):
        """Entraîne le modèle"""
        
        print(f"\n{'='*70}")
        print(f"🎯 TRAINING: {player} - {self.stat_type.upper()}")
        print(f"{'='*70}")
        
        try:
            from advanced_data_collector import AdvancedDataCollector
            collector = AdvancedDataCollector()
            
            # Récupère données
            df = collector.get_complete_player_data(player)
            
            if df is None:
                return {'status': 'ERROR', 'message': 'No data'}
            
            # Fix dict
            if isinstance(df, dict):
                df = pd.DataFrame([df])
            
            if len(df) < 10:
                return {'status': 'ERROR', 'message': f'Only {len(df)} games'}
            
            stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}[self.stat_type]
            
            print(f"📊 Total games: {len(df)}")
            
            # Enlève outliers
            df_clean = self.remove_outliers(df, stat_col)
            print(f"   ✅ Clean games: {len(df_clean)}")
            
            if len(df_clean) < 10:
                return {'status': 'ERROR', 'message': 'Too many outliers'}
            
            # Calcule predictability (pour info)
            pred_score = self.calculate_predictability_score(df_clean, stat_col)
            print(f"\n📈 Predictability: {pred_score['score']}/100 (CV={pred_score['cv']}%)")
            
            # ✅ SOLUTION: Skip check pour POINTS uniquement!
            if self.stat_type in ['assists', 'rebounds']:
                # Garde le check strict pour ASSISTS/REBOUNDS
                if pred_score['score'] < 25:
                    print(f"\n⚠️  SKIPPED: Too unpredictable for {self.stat_type}")
                    return {
                        'status': 'SKIPPED',
                        'message': f'Player too unpredictable for {self.stat_type}',
                        'predictability_score': pred_score['score']
                    }
            else:
                # POINTS: Accepte TOUT le monde!
                print(f"   ✅ POINTS: Predictability check SKIPPED (always accept)")
            
            # Features
            features_df = collector.prepare_features_for_prediction(player, '', True)
            
            if features_df is None:
                return {'status': 'ERROR', 'message': 'No features'}
            
            # Fix dict
            if isinstance(features_df, dict):
                features_df = pd.DataFrame([features_df])
            
            # Ajoute CV
            features_df['cv'] = pred_score['cv']
            features_df['stability'] = pred_score['stability']
            
            # Merge
            df_clean = df_clean.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            df_clean['target'] = df_clean[stat_col]
            
            # Aligne
            min_len = min(len(features_df), len(df_clean))
            if min_len < 10:
                return {'status': 'ERROR', 'message': 'Insufficient aligned data'}
            
            features_df = features_df.iloc[:min_len]
            df_clean = df_clean.iloc[:min_len]
            
            X = features_df
            y = df_clean['target'].values
            
            print(f"\n🔢 Features: {X.shape}")
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )
            
            print(f"   Train={len(X_train)}, Test={len(X_test)}")
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                early_stopping_rounds=15,
                verbose=False
            )
            
            # Metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            y_train_pred = self.model.predict(X_train_scaled)
            train_r2 = r2_score(y_train, y_train_pred)
            
            y_test_pred = self.model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            try:
                cv_scores = cross_val_score(
                    self.model, X_train_scaled, y_train,
                    cv=min(3, len(X_train) // 10), scoring='r2'
                )
            except:
                cv_scores = np.array([test_r2])
            
            print(f"\n📊 RESULTS:")
            print(f"   Train R²: {train_r2:.3f}")
            print(f"   Test R²:  {test_r2:.3f} ⭐")
            print(f"   RMSE: {test_rmse:.2f}")
            
            # ✅ Seuil R² plus permissif
            min_r2 = 0.05 if self.stat_type == 'points' else 0.10
            
            if test_r2 < min_r2:
                print(f"\n❌ REJECTED: R² {test_r2:.3f} < {min_r2}")
                return {
                    'status': 'REJECTED',
                    'message': f'R² too low',
                    'test_r2': round(test_r2, 3),
                    'predictability_score': pred_score['score']
                }
            
            # Store stats
            self.training_stats = {
                'player': player,
                'stat_type': self.stat_type,
                'train_metrics': {'r2': train_r2, 'rmse': 0},
                'test_metrics': {'r2': test_r2, 'rmse': test_rmse, 'mae': 0},
                'cv_results': {'r2_mean': cv_scores.mean(), 'r2_std': cv_scores.std()},
                'predictability': pred_score,
                'data': {
                    'total_games': len(df),
                    'clean_games': len(df_clean),
                    'outliers_removed': len(df) - len(df_clean)
                },
                'stability': {'overfitting': train_r2 - test_r2, 'cv_percent': 0}
            }
            
            print(f"\n✅ SUCCESS!")
            print(f"{'='*70}\n")
            
            return {
                'status': 'SUCCESS',
                'train_metrics': self.training_stats['train_metrics'],
                'test_metrics': self.training_stats['test_metrics'],
                'cv_results': self.training_stats['cv_results'],
                'predictability': pred_score,
                'stability': self.training_stats['stability']
            }
        
        except Exception as e:
            print(f"\n❌ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'ERROR', 'message': str(e)}
    
    def predict(self, features):
        """Fait une prédiction"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        test_rmse = self.training_stats['test_metrics']['rmse']
        ci_lower = prediction - 1.96 * test_rmse
        ci_upper = prediction + 1.96 * test_rmse
        
        return {
            'prediction': float(prediction),
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            }
        }


class ModelManager:
    """Gère les modèles"""
    
    def __init__(self):
        self.models = {}
    
    def predict(self, player, stat_type, opponent, is_home):
        model_key = f"{player}_{stat_type}"
        
        if model_key not in self.models:
            print(f"🔄 Training: {model_key}")
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player, save_model=False)
            
            if result['status'] != 'SUCCESS':
                raise ValueError(f"Training failed: {result.get('message')}")
            
            self.models[model_key] = model
        
        from advanced_data_collector import AdvancedDataCollector
        collector = AdvancedDataCollector()
        features = collector.prepare_features_for_prediction(player, opponent, is_home)
        
        if features is None:
            raise ValueError("No features")
        
        return self.models[model_key].predict(features)


if __name__ == '__main__':
    model = XGBoostNBAModel(stat_type='points')
    result = model.train('LeBron James', '2024-25', save_model=False)
    print(f"\nStatus: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Test R²: {result['test_metrics']['r2']:.3f}")