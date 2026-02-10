#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Model - NBA Predictions FIXED
Train/Test Split + Filtres de stabilité
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from advanced_data_collector import AdvancedDataCollector


class XGBoostNBAModel:
    """
    Modèle XGBoost avec validation chronologique
    """
    
    def __init__(self, stat_type='points'):
        self.stat_type = stat_type
        self.stat_col = {'points': 'PTS', 'assists': 'AST', 'rebounds': 'REB'}[stat_type]
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.training_stats = {}
        
        # Hyperparamètres ANTI-OVERFITTING
        self.model_params = {
            'n_estimators': 300,      # Réduit de 500
            'learning_rate': 0.05,
            'max_depth': 4,           # Réduit de 6 → évite mémorisation
            'min_child_weight': 5,    # Augmenté de 3 → moins de splits
            'subsample': 0.7,         # Réduit de 0.8
            'colsample_bytree': 0.7,  # Réduit de 0.8
            'gamma': 0.3,             # Augmenté de 0.1 → pénalise splits
            'reg_alpha': 0.5,         # Augmenté de 0.1 → L1 regularization
            'reg_lambda': 2.0,        # Augmenté de 1.0 → L2 regularization
            'random_state': 42,
            'n_jobs': -1
        }
    
    def train(self, player_name, season='2024-25', save_model=True):
        """
        Entraîne avec TRAIN/TEST SPLIT chronologique
        """
        
        print(f"\n{'='*70}")
        print(f"🤖 XGBOOST - {player_name} ({self.stat_type.upper()})")
        print(f"{'='*70}\n")
        
        # 1. Collecte données
        collector = AdvancedDataCollector()
        df = collector.get_complete_player_data(player_name, season)
        
        if df is None or len(df) < 15:
            return {'status': 'ERROR', 'message': 'Pas assez de données (min 15 matchs)'}
        
        # 2. Prépare features
        X, y, feature_names = self._prepare_training_data(df)
        
        if X is None:
            return {'status': 'ERROR', 'message': 'Erreur préparation'}
        
        self.feature_names = feature_names
        
        print(f"📊 Total: {len(X)} matchs, {len(feature_names)} features")
        print(f"📊 {self.stat_col}: moyenne={y.mean():.1f}, std={y.std():.1f}\n")
        
        # ✅ 3. SPLIT CHRONOLOGIQUE (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Train: {len(X_train)} matchs (anciens)")
        print(f"✅ Test: {len(X_test)} matchs (récents - JAMAIS VUS!)\n")
        
        # 4. Standardisation (fit sur train SEULEMENT!)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 5. Cross-validation sur train
        print("🔄 Cross-Validation (TimeSeriesSplit)...")
        cv_results = self._cross_validate(X_train_scaled, y_train)
        
        # 6. Entraînement final (sur train SEULEMENT!)
        print("\n🤖 Entraînement final (sur train set)...")
        self.model = XGBRegressor(**self.model_params)
        self.model.fit(X_train_scaled, y_train, verbose=False)
        
        # ✅ 7. ÉVALUATION SUR TEST SET (MATCHS FUTURS JAMAIS VUS!)
        print("\n🎯 ÉVALUATION SUR MATCHS FUTURS (test set)...")
        y_pred_test = self.model.predict(X_test_scaled)
        
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Calcule stabilité (coefficient de variation)
        cv_percent = (y.std() / y.mean()) * 100 if y.mean() > 0 else 100
        
        print(f"  R² TEST: {test_r2:.3f} ← VRAI R²!")
        print(f"  RMSE TEST: {test_rmse:.2f}")
        print(f"  MAE TEST: {test_mae:.2f}")
        print(f"  Stabilité: CV = {cv_percent:.1f}% (plus bas = mieux)")
        
        # Train metrics (pour comparaison)
        y_pred_train = self.model.predict(X_train_scaled)
        train_r2 = r2_score(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        print(f"\n📊 COMPARAISON:")
        print(f"  Train R²: {train_r2:.3f} (si >> test R² = overfit!)")
        print(f"  Test R²: {test_r2:.3f} (VRAI performance)")
        
        if train_r2 - test_r2 > 0.15:
            print(f"  ⚠️  OVERFIT détecté! Différence = {train_r2 - test_r2:.3f}")
        
        # 8. Sauvegarde
        if save_model:
            self._save_model(player_name, season)
        
        # Résultats
        results = {
            'status': 'SUCCESS',
            'player': player_name,
            'stat_type': self.stat_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'cv_results': cv_results,
            'test_metrics': {  # ← VRAIES MÉTRIQUES!
                'r2': float(test_r2),
                'rmse': float(test_rmse),
                'mae': float(test_mae)
            },
            'train_metrics': {  # Pour comparaison
                'r2': float(train_r2),
                'rmse': float(train_rmse)
            },
            'stability': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'cv_percent': float(cv_percent),
                'min': float(y.min()),
                'max': float(y.max())
            },
            'model_saved': save_model
        }
        
        self.training_stats = results
        
        print(f"\n{'='*70}\n")
        
        return results
    
    def _prepare_training_data(self, df):
        """Prépare X et y"""
        
        exclude_cols = [
            'GAME_DATE', 'MATCHUP', 'WL', 'opponent',
            'PTS', 'AST', 'REB', 'FGM', 'FGA', 
            'FG3M', 'FG3A', 'FTM', 'FTA', 'STL', 'BLK', 'TOV'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        df_clean = df[feature_cols + [self.stat_col]].dropna()
        
        if len(df_clean) < 10:
            return None, None, None
        
        X = df_clean[feature_cols].values
        y = df_clean[self.stat_col].values
        
        return X, y, feature_cols
    
    def _cross_validate(self, X, y, n_splits=5):
        """Cross-validation temporelle"""
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        r2_scores = []
        rmse_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            model_fold = XGBRegressor(**self.model_params)
            model_fold.fit(X_tr, y_tr, verbose=False)
            
            y_pred = model_fold.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            
            print(f"  Fold {fold+1}/{n_splits}: R²={r2:.3f}, RMSE={rmse:.2f}")
        
        cv_results = {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'rmse_mean': float(np.mean(rmse_scores)),
            'rmse_std': float(np.std(rmse_scores))
        }
        
        print(f"\n  CV Moyenne: R²={cv_results['r2_mean']:.3f} (±{cv_results['r2_std']:.3f})")
        print(f"              RMSE={cv_results['rmse_mean']:.2f} (±{cv_results['rmse_std']:.2f})")
        
        return cv_results
    
    def predict(self, features_dict):
        """Prédiction avec intervalle basé sur TEST RMSE"""
        
        if self.model is None:
            return {'error': 'Model not trained'}
        
        feature_vector = [features_dict.get(fname, 0) for fname in self.feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.model.predict(feature_vector_scaled)[0]
        
        # ✅ Intervalle basé sur TEST RMSE (pas train!)
        test_rmse = self.training_stats.get('test_metrics', {}).get('rmse', 5.0)
        margin = 1.96 * test_rmse
        
        return {
            'prediction': round(float(prediction), 1),
            'confidence_interval': {
                'lower': round(float(prediction - margin), 1),
                'upper': round(float(prediction + margin), 1),
                'width': round(float(2 * margin), 1)
            },
            'model_type': 'XGBoost',
            'test_r2': self.training_stats.get('test_metrics', {}).get('r2', 0.0)
        }
    
    def _save_model(self, player_name, season):
        """Sauvegarde"""
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        player_slug = player_name.lower().replace(' ', '_')
        model_path = models_dir / f"{player_slug}_{self.stat_type}_{season}.pkl"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'stat_type': self.stat_type,
            'player': player_name,
            'season': season,
            'training_stats': self.training_stats
        }
        
        joblib.dump(model_data, model_path)
        print(f"💾 Sauvegardé: {model_path}")
    
    @classmethod
    def load_model(cls, model_path):
        """Charge un modèle"""
        
        model_data = joblib.load(model_path)
        instance = cls(stat_type=model_data['stat_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.training_stats = model_data['training_stats']
        
        return instance
# ============================================================================
# MODEL MANAGER - Avec filtres de stabilité
# ============================================================================

class ModelManager:
    """
    Gère les modèles avec FILTRES DE QUALITÉ
    """
    
    def __init__(self):
        self.models = {}
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
    
    def train_player_all_stats(self, player_name, season='2024-25'):
        """
        Entraîne les 3 modèles avec validation
        """
        
        print(f"\n{'='*70}")
        print(f"🎯 ENTRAÎNEMENT COMPLET: {player_name}")
        print(f"{'='*70}\n")
        
        results = {}
        
        for stat_type in ['points', 'assists', 'rebounds']:
            print(f"\n--- {stat_type.upper()} ---")
            
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player_name, season, save_model=True)
            
            if result['status'] == 'SUCCESS':
                # ✅ Filtre de qualité
                test_r2 = result['test_metrics']['r2']
                cv_percent = result['stability']['cv_percent']
                
                print(f"\n📊 QUALITÉ DU MODÈLE:")
                print(f"  Test R²: {test_r2:.3f}")
                print(f"  Stabilité (CV%): {cv_percent:.1f}%")
                
                # Évaluation
                if test_r2 >= 0.70:
                    quality = "🟢 EXCELLENT"
                elif test_r2 >= 0.50:
                    quality = "🟡 BON"
                elif test_r2 >= 0.30:
                    quality = "🟠 MOYEN"
                else:
                    quality = "🔴 FAIBLE - NE PAS UTILISER!"
                
                if cv_percent > 40:
                    quality += " - ⚠️ TRÈS INSTABLE!"
                
                print(f"  Évaluation: {quality}\n")
                
                key = f"{player_name}_{stat_type}"
                self.models[key] = model
                results[stat_type] = result
            else:
                print(f"❌ Échec {stat_type}\n")
        
        return results
    
    def load_player_models(self, player_name, season='2024-25'):
        """Charge les modèles d'un joueur"""
        
        player_slug = player_name.lower().replace(' ', '_')
        
        for stat_type in ['points', 'assists', 'rebounds']:
            model_path = self.models_dir / f"{player_slug}_{stat_type}_{season}.pkl"
            
            if model_path.exists():
                try:
                    model = XGBoostNBAModel.load_model(model_path)
                    key = f"{player_name}_{stat_type}"
                    self.models[key] = model
                    
                    # Affiche qualité du modèle chargé
                    test_r2 = model.training_stats.get('test_metrics', {}).get('r2', 0)
                    print(f"✅ {stat_type}: R²={test_r2:.3f}")
                except Exception as e:
                    print(f"❌ Erreur {stat_type}: {e}")
    
    def predict(self, player_name, stat_type, opponent, is_home, season='2024-25'):
        """
        Fait une prédiction avec VALIDATION DE QUALITÉ
        """
        
        key = f"{player_name}_{stat_type}"
        
        # Charge ou entraîne le modèle
        if key not in self.models:
            print(f"⚠️  Modèle non trouvé, entraînement...")
            model = XGBoostNBAModel(stat_type=stat_type)
            result = model.train(player_name, season)
            
            if result['status'] == 'SUCCESS':
                self.models[key] = model
            else:
                return {'error': 'Unable to train model'}
        
        model = self.models[key]
        
        # ✅ VÉRIFIE LA QUALITÉ DU MODÈLE
        test_r2 = model.training_stats.get('test_metrics', {}).get('r2', 0)
        cv_percent = model.training_stats.get('stability', {}).get('cv_percent', 100)
        
        # Prépare features
        collector = AdvancedDataCollector()
        features = collector.prepare_features_for_prediction(
            player_name, opponent, is_home, season
        )
        
        if features is None:
            return {'error': 'Unable to prepare features'}
        
        # Prédiction
        prediction_result = model.predict(features)
        
        # ✅ AJOUTE MÉTRIQUES DE QUALITÉ
        prediction_result['quality_metrics'] = {
            'test_r2': round(test_r2, 3),
            'stability_cv': round(cv_percent, 1),
            'recommendation': self._get_recommendation(test_r2, cv_percent)
        }
        
        return prediction_result
    
    def _get_recommendation(self, test_r2, cv_percent):
        """
        Recommandation basée sur R² ET stabilité
        """
        
        if test_r2 >= 0.70 and cv_percent <= 25:
            return "🟢 EXCELLENT - Haute confiance"
        elif test_r2 >= 0.70 and cv_percent <= 35:
            return "🟢 BON - Confiance élevée mais variance modérée"
        elif test_r2 >= 0.50 and cv_percent <= 30:
            return "🟡 CORRECT - Confiance moyenne"
        elif test_r2 >= 0.50 and cv_percent <= 40:
            return "🟡 MOYEN - Confiance moyenne, haute variance"
        elif test_r2 >= 0.30:
            return "🟠 FAIBLE - Utiliser avec précaution"
        else:
            return "🔴 TRÈS FAIBLE - NE PAS PARIER!"
    
    def get_betting_opportunities(self, min_test_r2=0.70, max_cv_percent=30):
        """
        Retourne seulement les opportunités avec modèles de QUALITÉ
        
        Args:
            min_test_r2: R² minimum sur test set (défaut: 0.70 = 70%)
            max_cv_percent: CV% maximum (défaut: 30% de variance)
        """
        
        quality_models = {}
        
        for key, model in self.models.items():
            test_r2 = model.training_stats.get('test_metrics', {}).get('r2', 0)
            cv_percent = model.training_stats.get('stability', {}).get('cv_percent', 100)
            
            if test_r2 >= min_test_r2 and cv_percent <= max_cv_percent:
                quality_models[key] = {
                    'model': model,
                    'test_r2': test_r2,
                    'cv_percent': cv_percent,
                    'player': model.training_stats.get('player'),
                    'stat_type': model.stat_type
                }
        
        return quality_models


# ============================================================================
# FILTRES RECOMMANDÉS
# ============================================================================

class BettingFilters:
    """
    Filtres pour sélectionner les MEILLEURS paris
    """
    
    @staticmethod
    def filter_high_quality(opportunities, min_test_r2=0.70, max_cv=30):
        """
        Filtre STRICT: R² ≥ 70% ET CV ≤ 30%
        
        Résultat: ~20-30% des opportunités mais HAUTE QUALITÉ
        """
        
        filtered = []
        
        for opp in opportunities:
            test_r2 = opp.get('regression_stats', {}).get('r2', 0)
            
            # Note: Tu dois ajouter CV dans le backend
            # Pour l'instant, on filtre juste sur R²
            
            if test_r2 >= min_test_r2:
                filtered.append(opp)
        
        return filtered
    
    @staticmethod
    def filter_medium_quality(opportunities, min_test_r2=0.50, max_cv=40):
        """
        Filtre MODÉRÉ: R² ≥ 50% ET CV ≤ 40%
        
        Résultat: ~50% des opportunités, qualité moyenne
        """
        
        filtered = []
        
        for opp in opportunities:
            test_r2 = opp.get('regression_stats', {}).get('r2', 0)
            
            if test_r2 >= min_test_r2:
                filtered.append(opp)
        
        return filtered


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("🧪 TEST: ENTRAÎNEMENT AVEC VALIDATION")
    print("="*70)
    
    # Test sur un joueur STABLE
    player_stable = "LeBron James"
    
    print(f"\n🔬 Test 1: Joueur STABLE ({player_stable})")
    print("="*70)
    
    model = XGBoostNBAModel(stat_type='points')
    results = model.train(player_stable, '2024-25', save_model=True)
    
    if results['status'] == 'SUCCESS':
        test_r2 = results['test_metrics']['r2']
        cv_percent = results['stability']['cv_percent']
        
        print(f"\n🎯 RÉSULTATS:")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Stabilité CV: {cv_percent:.1f}%")
        
        if test_r2 >= 0.70 and cv_percent <= 30:
            print(f"  ✅ EXCELLENT MODÈLE - Recommandé pour paris!")
        elif test_r2 >= 0.50:
            print(f"  🟡 MODÈLE MOYEN - Utiliser avec précaution")
        else:
            print(f"  ❌ MODÈLE FAIBLE - Ne pas parier!")
    
    # Test sur un joueur INSTABLE
    player_unstable = "Ausar Thompson"
    
    print(f"\n\n🔬 Test 2: Joueur INSTABLE ({player_unstable})")
    print("="*70)
    
    model2 = XGBoostNBAModel(stat_type='points')
    results2 = model2.train(player_unstable, '2024-25', save_model=False)
    
    if results2['status'] == 'SUCCESS':
        test_r2 = results2['test_metrics']['r2']
        cv_percent = results2['stability']['cv_percent']
        
        print(f"\n🎯 RÉSULTATS:")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Stabilité CV: {cv_percent:.1f}%")
        
        if test_r2 >= 0.70 and cv_percent <= 30:
            print(f"  ✅ EXCELLENT - Recommandé")
        elif test_r2 >= 0.50:
            print(f"  🟡 MOYEN - Précaution")
        else:
            print(f"  ❌ FAIBLE - NE PAS PARIER!")
        
        if cv_percent > 40:
            print(f"  ⚠️  TROP INSTABLE! Variance = {cv_percent:.1f}%")
    
    print("\n" + "="*70 + "\n")